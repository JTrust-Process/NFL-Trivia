import os, time, json, random
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import pandas as pd

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature

from src.models import db, User, GameSession, QuestionRecord, Season, SeasonEntry, UserAchievement, ACHIEVEMENTS
from src.trivia import (
    load_questions, choose_next, update_skill, check_answer,
    generate_choices, SessionState,
)
from src.gemini_ai import (
    smarter_difficulty_calibration,
    auto_generate_questions,
    get_personalized_weights,
    get_fun_fact,
    get_explanation,
)

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24))

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///nfl_trivia_dev.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

# ── Email config (Flask-Mail) ─────────────────────────────────────────────────
app.config["MAIL_SERVER"]   = "smtp.gmail.com"
app.config["MAIL_PORT"]     = 587
app.config["MAIL_USE_TLS"]  = True
app.config["MAIL_USERNAME"] = os.environ.get("MAIL_USERNAME")
app.config["MAIL_PASSWORD"] = os.environ.get("MAIL_PASSWORD")
app.config["MAIL_DEFAULT_SENDER"] = os.environ.get("MAIL_USERNAME")
mail = Mail(app)
serializer = URLSafeTimedSerializer(app.secret_key)

login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message = "Please log in to play."

DATA_CSV = "data/nfl_trivia.csv"

# ── Question cache ────────────────────────────────────────────────────────────
# Avoids reading the CSV from disk on every single API call.
# Invalidated automatically when the CSV is written to (auto-generate, calibration).
_question_cache: list = []
_cache_mtime: float = 0.0

def get_questions() -> list:
    """Return cached questions, reloading from CSV only if the file has changed.
    Filters out pending and rejected questions so only approved ones are served."""
    global _question_cache, _cache_mtime
    try:
        mtime = os.path.getmtime(DATA_CSV)
        if mtime != _cache_mtime or not _question_cache:
            all_qs = load_questions(DATA_CSV)
            # Read status column directly to filter — load_questions doesn't expose it
            df = pd.read_csv(DATA_CSV)
            if "status" in df.columns:
                approved_ids = set(df[df["status"] == "approved"]["id"].tolist())
                _question_cache = [q for q in all_qs if q.id in approved_ids]
            else:
                _question_cache = all_qs  # no status col yet, serve all
            _cache_mtime = mtime
    except FileNotFoundError:
        _question_cache = []
    return _question_cache


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


def init_db():
    with app.app_context():
        db.create_all()


def get_or_create_season():
    """Get the current active season, or create a new one for this week."""
    from datetime import date, timedelta
    today = date.today()
    # Week starts Monday
    week_start = today - timedelta(days=today.weekday())
    week_end   = week_start + timedelta(days=6)

    season = Season.query.filter_by(week_start=week_start).first()
    if not season:
        # Close out any previously active seasons
        Season.query.filter_by(is_active=True).update({"is_active": False})
        season = Season(week_start=week_start, week_end=week_end, is_active=True)
        db.session.add(season)
        db.session.commit()
    return season


def check_and_award_achievements(user_id, game_session, all_sessions, cat_records):
    """
    Checks all achievement conditions after a game ends.
    Returns list of newly earned achievement keys.
    """
    already_earned = {a.achievement for a in
                      UserAchievement.query.filter_by(user_id=user_id).all()}
    new_achievements = []

    def award(key):
        if key not in already_earned:
            db.session.add(UserAchievement(user_id=user_id, achievement=key))
            new_achievements.append(key)
            already_earned.add(key)

    total_games  = len(all_sessions)
    best_mu      = max((s.final_mu for s in all_sessions), default=0)
    best_streak  = max((s.best_streak for s in all_sessions), default=0)

    # First Win
    if total_games >= 1:
        award("first_win")

    # Streak badges
    if best_streak >= 5:
        award("on_fire")
    if best_streak >= 10:
        award("unstoppable")

    # Sharp — 90%+ accuracy in this game
    if game_session.accuracy >= 0.90:
        award("sharp")

    # Grinder / Century
    if total_games >= 10:
        award("grinder")
    if total_games >= 100:
        award("century")

    # μ milestones
    if best_mu >= 3.0:
        award("high_iq")
    if best_mu >= 5.0:
        award("legend")

    # Category Master — 80%+ in any category with 10+ questions
    cat_totals  = defaultdict(int)
    cat_correct = defaultdict(int)
    for r in cat_records:
        cat_totals[r.category]  += 1
        if r.is_correct:
            cat_correct[r.category] += 1
    for cat, total in cat_totals.items():
        if total >= 10 and cat_correct[cat] / total >= 0.80:
            award("category_master")
            break

    if new_achievements:
        db.session.commit()

    return new_achievements


# ── Auth ──────────────────────────────────────────────────────────────────────
@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        email    = request.form.get("email", "").strip().lower() or None
        if not username or not password:
            flash("Username and password are required.", "error")
        elif len(username) < 3:
            flash("Username must be at least 3 characters.", "error")
        elif len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
        elif User.query.filter_by(username=username).first():
            flash("That username is taken.", "error")
        elif email and User.query.filter_by(email=email).first():
            flash("That email is already registered.", "error")
        else:
            user = User(username=username, email=email)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            login_user(user)
            return redirect(url_for("index"))
    return render_template("auth.html", mode="register")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        user = User.query.filter_by(username=username).first()
        if not user or not user.check_password(password):
            flash("Invalid username or password.", "error")
        else:
            login_user(user, remember=True)
            return redirect(url_for("index"))
    return render_template("auth.html", mode="login")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email    = request.form.get("email", "").strip().lower()
        user = User.query.filter_by(username=username).first()
        # Always show the same message to prevent username enumeration
        if user and user.email and user.email.lower() == email:
            try:
                token = serializer.dumps(user.id, salt="pw-reset")
                reset_url = url_for("reset_password", token=token, _external=True)
                msg = Message("Reset your NFL Trivia password", recipients=[email])
                msg.body = (
                    f"Hi {username},\n\n"
                    f"Click the link below to reset your password (expires in 1 hour):\n\n"
                    f"{reset_url}\n\n"
                    f"If you didn't request this, just ignore this email.\n\n"
                    f"— NFL Trivia"
                )
                mail.send(msg)
            except Exception as e:
                print(f"[Mail] Failed to send reset email: {e}")
        flash("If that username and email match an account, you'll get a reset link shortly.", "success")
        return redirect(url_for("forgot_password"))
    return render_template("forgot_password.html")


@app.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token):
    try:
        user_id = serializer.loads(token, salt="pw-reset", max_age=3600)
    except (SignatureExpired, BadSignature):
        flash("This reset link is invalid or has expired. Please request a new one.", "error")
        return redirect(url_for("forgot_password"))

    user = db.session.get(User, user_id)
    if not user:
        flash("User not found.", "error")
        return redirect(url_for("forgot_password"))

    if request.method == "POST":
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm", "")
        if len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
        elif password != confirm:
            flash("Passwords don't match.", "error")
        else:
            user.set_password(password)
            db.session.commit()
            flash("Password updated! You can now log in.", "success")
            return redirect(url_for("login"))
    return render_template("reset_password.html", token=token)


# ── Pages ─────────────────────────────────────────────────────────────────────
@app.route("/")
@login_required
def index():
    recent = (GameSession.query
              .filter_by(user_id=current_user.id)
              .order_by(GameSession.played_at.desc())
              .limit(5).all())
    return render_template("index.html", user=current_user, recent=recent)


@app.route("/leaderboard")
@login_required
def leaderboard_page():
    rows = (db.session.query(
                User.username,
                db.func.max(GameSession.final_mu).label("best_mu"),
                db.func.avg(GameSession.accuracy).label("avg_accuracy"),
                db.func.max(GameSession.best_streak).label("best_streak"),
                db.func.count(GameSession.id).label("games_played"),
            )
            .join(GameSession, GameSession.user_id == User.id)
            .group_by(User.id, User.username)
            .order_by(db.desc("best_mu"), db.desc("avg_accuracy"))
            .limit(20).all())
    board = [{"username": r.username, "best_mu": round(r.best_mu, 2),
              "avg_accuracy": round((r.avg_accuracy or 0) * 100, 1),
              "best_streak": r.best_streak, "games_played": r.games_played}
             for r in rows]
    return render_template("leaderboard.html", board=board, user=current_user)


@app.route("/leaderboard/weekly")
@login_required
def weekly_leaderboard():
    season = get_or_create_season()
    rows = (db.session.query(
                User.username,
                SeasonEntry.best_mu,
                SeasonEntry.games,
                SeasonEntry.accuracy,
            )
            .join(SeasonEntry, SeasonEntry.user_id == User.id)
            .filter(SeasonEntry.season_id == season.id)
            .order_by(db.desc(SeasonEntry.best_mu))
            .limit(20).all())
    board = [{"username": r.username, "best_mu": round(r.best_mu, 2),
              "games": r.games, "avg_accuracy": round(r.accuracy * 100, 1)}
             for r in rows]
    return render_template("leaderboard_weekly.html", board=board,
                           season=season, user=current_user)
@login_required
def profile():
    sessions = (GameSession.query
                .filter_by(user_id=current_user.id)
                .order_by(GameSession.played_at.desc())
                .limit(20).all())
    records = (QuestionRecord.query.join(GameSession)
               .filter(GameSession.user_id == current_user.id).all())
    cat_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in records:
        cat_stats[r.category]["total"] += 1
        if r.is_correct:
            cat_stats[r.category]["correct"] += 1
    cat_data = [{"category": c, "pct": round(v["correct"] / v["total"] * 100),
                 "correct": v["correct"], "total": v["total"]}
                for c, v in sorted(cat_stats.items(), key=lambda x: -x[1]["correct"])
                if v["total"] > 0]

    # Achievements
    earned = {a.achievement: a.earned_at for a in
              UserAchievement.query.filter_by(user_id=current_user.id).all()}
    earned_achievements = [
        {"key": k, **ACHIEVEMENTS[k], "earned_at": earned.get(k),
         "unlocked": k in earned}
        for k in ACHIEVEMENTS
    ]

    # μ history for career graph (last 20 games, chronological)
    history_sessions = (GameSession.query
                        .filter_by(user_id=current_user.id)
                        .order_by(GameSession.played_at.asc())
                        .limit(20).all())
    mu_history = [{"mu": round(s.final_mu, 2),
                   "date": s.played_at.strftime("%b %d")} for s in history_sessions]
    return render_template("profile.html", user=current_user,
                           sessions=sessions, cat_data=cat_data,
                           achievements=earned_achievements,
                           all_achievements=ACHIEVEMENTS,
                           mu_history=mu_history)


# ── Game API ──────────────────────────────────────────────────────────────────
@app.route("/api/start", methods=["POST"])
@login_required
def start_game():
    data  = request.json
    daily = data.get("daily", False)
    cat   = data.get("category", "all")

    if daily:
        import hashlib
        seed = int(hashlib.md5(datetime.now(timezone.utc).strftime("%Y-%m-%d").encode()).hexdigest(), 16) % (2**31)
        random.seed(seed)
        rounds = 10
    else:
        rounds = int(data.get("rounds", 15))

    # ── Feature 3: Load personalized weights for this player ──────────────────
    # Pull their all-time category performance from DB to bias question selection
    all_records = (QuestionRecord.query
                   .join(GameSession)
                   .filter(GameSession.user_id == current_user.id)
                   .all())
    cat_counts  = defaultdict(int)
    cat_correct = defaultdict(int)
    for r in all_records:
        if r.category:
            cat_counts[r.category] += 1
            if r.is_correct:
                cat_correct[r.category] += 1

    personalized_weights = get_personalized_weights(
        dict(cat_counts), dict(cat_correct)
    )

    session.clear()
    session["game"] = {
        "mode": data.get("mode", "free"),
        "timed": data.get("timed", False),
        "rounds": rounds,
        "multiplayer": data.get("multiplayer", False),
        "daily": daily,
        "category": cat,
        "difficulty": data.get("difficulty", "all"),
        "current_round": 0, "asked_ids": [], "records": [], "hint_used": False,
        "q_start_time": None,
        "personalized_weights": personalized_weights,
        "p1_name": current_user.username,
        "p1_mu": 0.0, "p1_streak": 0, "p1_best_streak": 0,
        "p1_streak_multiplier": 1.0, "p1_cat_counts": {}, "p1_cat_correct": {},
        "p2_name": data.get("player2_name", "Player 2"),
        "p2_mu": 0.0, "p2_streak": 0, "p2_best_streak": 0,
        "p2_streak_multiplier": 1.0, "p2_cat_counts": {}, "p2_cat_correct": {},
        "active_player": 1,
    }
    return jsonify({"status": "ok", "rounds": rounds, "mode": data.get("mode", "free"),
                    "p1_name": current_user.username})


@app.route("/api/question", methods=["GET"])
@login_required
def get_question():
    g = session.get("game")
    if not g:
        return jsonify({"error": "No active game"}), 400
    if g["current_round"] >= g["rounds"]:
        return jsonify({"game_over": True})

    questions = get_questions()

    # Filter by category if set
    cat = g.get("category", "all")
    if cat and cat != "all":
        questions = [q for q in questions if q.category == cat]

    # Filter by difficulty if set
    diff = g.get("difficulty", "all")
    if diff and diff != "all":
        try:
            diff_val = int(diff)
            questions = [q for q in questions if round(q.difficulty) == diff_val]
        except ValueError:
            pass

    player = g["active_player"]
    state  = _build_state(g, player)

    # Pass personalized weights into choose_next
    q = choose_next(state, questions, set(g["asked_ids"]),
                    personalized_weights=g.get("personalized_weights"))
    if q is None:
        return jsonify({"game_over": True})

    g["asked_ids"].append(q.id)
    g["hint_used"] = False
    g["q_start_time"] = time.time()
    session["game"] = g

    # ── Feature 2: Check if pool is running low, generate more in background ──
    auto_generate_questions(DATA_CSV, g["asked_ids"], current_user.id)

    choices = generate_choices(q, questions) if g["mode"] == "multiple_choice" else None
    return jsonify({
        "id": q.id, "question": q.question, "category": q.category,
        "difficulty": round(q.difficulty, 1), "choices": choices,
        "round": g["current_round"] + 1, "rounds": g["rounds"],
        "timed": g["timed"], "time_limit": 15 if g["timed"] else None,
        "multiplayer": g["multiplayer"], "active_player": player,
        "active_name": g[f"p{player}_name"],
        "p1_mu": round(g["p1_mu"], 2),
        "p2_mu": round(g["p2_mu"], 2) if g["multiplayer"] else None,
        "p1_streak": g["p1_streak"],
        "p2_streak": g["p2_streak"] if g["multiplayer"] else None,
    })


@app.route("/api/hint", methods=["GET"])
@login_required
def get_hint():
    g = session.get("game")
    if not g or g.get("hint_used") or not g["asked_ids"]:
        return jsonify({"hint": ""})
    q = next((x for x in get_questions() if x.id == g["asked_ids"][-1]), None)
    hint_text = (q.hint if q and q.hint and q.hint != "nan" else "No hint available.")
    player = g["active_player"]
    g[f"p{player}_mu"] = round(g[f"p{player}_mu"] - 0.15, 4)
    g["hint_used"] = True
    session["game"] = g
    return jsonify({"hint": hint_text, "penalty": -0.15})


@app.route("/api/answer", methods=["POST"])
@login_required
def submit_answer():
    g = session.get("game")
    if not g or not g["asked_ids"]:
        return jsonify({"error": "No active game"}), 400
    data      = request.json
    user_ans  = data.get("answer", "").strip()
    timed_out = data.get("timed_out", False)
    q = next((x for x in get_questions() if x.id == g["asked_ids"][-1]), None)
    if not q:
        return jsonify({"error": "Question not found"}), 400
    t_elapsed = min(time.time() - (g["q_start_time"] or time.time()), 60.0)
    if timed_out:
        is_correct, feedback = False, f"Time's up! The answer was: {q.answer}"
    else:
        is_correct, feedback = check_answer(user_ans, q.answer)
    player     = g["active_player"]
    prev_mu    = g[f"p{player}_mu"]
    new_mu     = round(update_skill(prev_mu, q.difficulty, is_correct, t_elapsed,
                                    g[f"p{player}_streak_multiplier"]), 4)
    if is_correct:
        g[f"p{player}_streak"] += 1
        g[f"p{player}_best_streak"] = max(g[f"p{player}_best_streak"], g[f"p{player}_streak"])
    else:
        g[f"p{player}_streak"] = 0
    g[f"p{player}_streak_multiplier"] = min(2.0, 1.0 + g[f"p{player}_streak"] * 0.25)
    g[f"p{player}_mu"] = new_mu
    cc = g[f"p{player}_cat_counts"]
    cr = g[f"p{player}_cat_correct"]
    cc[q.category] = cc.get(q.category, 0) + 1
    if is_correct:
        cr[q.category] = cr.get(q.category, 0) + 1
    g[f"p{player}_cat_counts"]  = cc
    g[f"p{player}_cat_correct"] = cr
    g["records"].append({"question_id": q.id, "category": q.category,
                         "difficulty": q.difficulty, "is_correct": int(is_correct),
                         "response_time": round(t_elapsed, 2), "mu_before": round(prev_mu, 4),
                         "mu_after": new_mu, "hint_used": g["hint_used"], "player": player})
    g["current_round"] += 1
    if g["multiplayer"]:
        g["active_player"] = 2 if player == 1 else 1
    session["game"] = g
    streak = g[f"p{player}_streak"]
    streak_msg = ({3:"🔥 3 in a row!",5:"🔥🔥 5 streak!",7:"🔥🔥🔥 Unstoppable!"}.get(streak,"")
                  or ("🔥🔥🔥 Unstoppable!" if streak > 7 else ""))

    # Fun fact — only on wrong answers, rate limited per user
    fun_fact = ""
    if not is_correct:
        fun_fact = get_fun_fact(q.question, q.answer, q.category, current_user.id)

    # Explanation — after every answer, rate limited per user
    explanation = get_explanation(q.question, q.answer, q.category,
                                  is_correct, current_user.id)

    return jsonify({
        "is_correct": is_correct, "feedback": feedback, "correct_answer": q.answer,
        "fun_fact": fun_fact,
        "explanation": explanation,
        "mu_before": round(prev_mu, 2), "mu_after": round(new_mu, 2),
        "delta": round(new_mu - prev_mu, 3), "streak": streak, "streak_msg": streak_msg,
        "multiplier": round(g[f"p{player}_streak_multiplier"], 2),
        "game_over": g["current_round"] >= g["rounds"], "active_player": player,
        "active_name": g[f"p{player}_name"],
        "p1_mu": round(g["p1_mu"], 2),
        "p2_mu": round(g["p2_mu"], 2) if g["multiplayer"] else None,
    })


@app.route("/api/results", methods=["POST"])
@login_required
def save_results():
    g = session.get("game")
    if not g:
        return jsonify({"error": "No active game"}), 400
    p1_records = [r for r in g["records"] if r["player"] == 1]
    p1_acc = sum(r["is_correct"] for r in p1_records) / len(p1_records) if p1_records else 0
    gs = GameSession(user_id=current_user.id, mode=g["mode"], timed=g["timed"],
                     multiplayer=g["multiplayer"], rounds=g["rounds"],
                     final_mu=g["p1_mu"], accuracy=p1_acc, best_streak=g["p1_best_streak"])
    db.session.add(gs)
    db.session.flush()
    for r in p1_records:
        db.session.add(QuestionRecord(
            session_id=gs.id, question_id=r["question_id"], category=r["category"],
            difficulty=r["difficulty"], is_correct=bool(r["is_correct"]),
            response_time=r["response_time"], mu_before=r["mu_before"],
            mu_after=r["mu_after"], hint_used=r["hint_used"], player=r["player"]))
    db.session.commit()

    # ── Update season entry ───────────────────────────────────────────────────
    try:
        season = get_or_create_season()
        entry  = SeasonEntry.query.filter_by(
            season_id=season.id, user_id=current_user.id).first()
        if not entry:
            entry = SeasonEntry(season_id=season.id, user_id=current_user.id)
            db.session.add(entry)
        entry.best_mu  = max(entry.best_mu, g["p1_mu"])
        entry.games   += 1
        # Running average accuracy
        entry.accuracy = ((entry.accuracy * (entry.games - 1)) + p1_acc) / entry.games
        db.session.commit()
    except Exception as e:
        print(f"[Season] Entry update failed: {e}")
    all_records = [{"question_id": r.question_id, "is_correct": r.is_correct,
                    "mu_before": r.mu_before} for r in QuestionRecord.query.all()]
    retrained = smarter_difficulty_calibration(DATA_CSV, all_records)

    rows = (db.session.query(User.username,
                db.func.max(GameSession.final_mu).label("best_mu"),
                db.func.avg(GameSession.accuracy).label("avg_accuracy"),
                db.func.max(GameSession.best_streak).label("best_streak"))
            .join(GameSession, GameSession.user_id == User.id)
            .group_by(User.id, User.username)
            .order_by(db.desc("best_mu")).limit(10).all())
    leaderboard = [{"username": r.username, "best_mu": round(r.best_mu, 2),
                    "accuracy": round((r.avg_accuracy or 0) * 100, 1),
                    "best_streak": r.best_streak} for r in rows]
    cats = {}
    for r in p1_records:
        c = r["category"]
        if c not in cats:
            cats[c] = {"correct": 0, "total": 0}
        cats[c]["total"] += 1
        if r["is_correct"]:
            cats[c]["correct"] += 1
    for c in cats:
        cats[c]["pct"] = round(cats[c]["correct"] / cats[c]["total"] * 100)
    winner = (g["p1_name"] if not g["multiplayer"] or g["p1_mu"] >= g["p2_mu"] else g["p2_name"])
    p2_result = None
    if g["multiplayer"]:
        p2_records = [r for r in g["records"] if r["player"] == 2]
        p2_acc = sum(r["is_correct"] for r in p2_records) / len(p2_records) if p2_records else 0
        p2_result = {"name": g["p2_name"], "mu": round(g["p2_mu"], 2),
                     "accuracy": round(p2_acc * 100, 1)}

    # ── Check achievements ────────────────────────────────────────────────────
    all_sessions = GameSession.query.filter_by(user_id=current_user.id).all()
    all_cat_records = QuestionRecord.query.join(GameSession).filter(
        GameSession.user_id == current_user.id).all()
    new_badges = check_and_award_achievements(
        current_user.id, gs, all_sessions, all_cat_records)
    new_badges_data = [{"key": k, **ACHIEVEMENTS[k]} for k in new_badges]

    return jsonify({"winner": winner, "retrained": retrained,
                    "p1": {"name": g["p1_name"], "mu": round(g["p1_mu"], 2),
                           "accuracy": round(p1_acc * 100, 1), "best_streak": g["p1_best_streak"]},
                    "p2": p2_result, "categories": cats, "leaderboard": leaderboard,
                    "new_badges": new_badges_data})


@app.route("/api/categories", methods=["GET"])
@login_required
def get_categories():
    questions = get_questions()
    cats = sorted(set(q.category for q in questions))
    return jsonify({"categories": cats})


@app.route("/submit-question", methods=["GET", "POST"])
@login_required
def submit_question():
    if request.method == "POST":
        question = request.form.get("question", "").strip()
        answer   = request.form.get("answer", "").strip()
        category = request.form.get("category", "").strip()
        hint     = request.form.get("hint", "").strip()
        try:
            diff = float(request.form.get("difficulty", 2.5))
            diff = round(max(1.0, min(5.0, diff)), 1)
        except ValueError:
            diff = 2.5

        if not question or not answer or not category:
            flash("Question, answer and category are required.", "error")
            return redirect(url_for("submit_question"))

        try:
            df = pd.read_csv(DATA_CSV)
            if "status" not in df.columns:
                df["status"] = "approved"
            if "source" not in df.columns:
                df["source"] = "manual"
            new_id = int(df["id"].max()) + 1 if len(df) else 1
            new_row = {
                "id": new_id, "question": question, "answer": answer,
                "difficulty": diff, "category": category,
                "hint": hint, "status": "pending", "source": "player",
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(DATA_CSV, index=False)
            global _cache_mtime
            _cache_mtime = 0.0
            flash("Thanks! Your question has been submitted for review.", "success")
        except Exception as e:
            print(f"[Submit] Error: {e}")
            flash("Something went wrong. Please try again.", "error")

        return redirect(url_for("submit_question"))
    return render_template("submit_question.html", user=current_user)


def _build_state(g, player):
    s = SessionState()
    s.mu = g[f"p{player}_mu"]
    s.streak = g[f"p{player}_streak"]
    s.best_streak = g[f"p{player}_best_streak"]
    s.streak_multiplier = g[f"p{player}_streak_multiplier"]
    s.category_counts  = defaultdict(int, g.get(f"p{player}_cat_counts", {}))
    s.category_correct = defaultdict(int, g.get(f"p{player}_cat_correct", {}))
    return s


# ── Admin ─────────────────────────────────────────────────────────────────────
def admin_required(f):
    """Decorator: requires login + is_admin flag."""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash("Admin access required.", "error")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


@app.route("/admin")
@login_required
@admin_required
def admin_page():
    """Shows all questions with pending/approved/rejected status."""
    df = pd.read_csv(DATA_CSV)
    if "status" not in df.columns:
        df["status"] = "approved"   # treat all existing questions as approved
        df["source"] = df.get("source", "manual")
    questions = df.to_dict("records")
    pending   = [q for q in questions if str(q.get("status","approved")) == "pending"]
    approved  = [q for q in questions if str(q.get("status","approved")) == "approved"]
    rejected  = [q for q in questions if str(q.get("status","approved")) == "rejected"]
    return render_template("admin.html", user=current_user,
                           pending=pending, approved=approved, rejected=rejected,
                           total=len(questions))


@app.route("/admin/review", methods=["POST"])
@login_required
@admin_required
def admin_review():
    """Handle approve / reject / edit actions from the admin page."""
    q_id     = request.form.get("id", type=int)
    action   = request.form.get("action")          # approve | reject | edit
    new_q    = request.form.get("question", "").strip()
    new_a    = request.form.get("answer", "").strip()
    new_diff = request.form.get("difficulty", type=float)
    new_hint = request.form.get("hint", "").strip()

    if not q_id or action not in ("approve", "reject", "edit"):
        flash("Invalid request.", "error")
        return redirect(url_for("admin_page"))

    df = pd.read_csv(DATA_CSV)
    if "status" not in df.columns:
        df["status"] = "approved"
    if "source" not in df.columns:
        df["source"] = "manual"

    mask = df["id"] == q_id
    if not mask.any():
        flash("Question not found.", "error")
        return redirect(url_for("admin_page"))

    if action == "approve":
        df.loc[mask, "status"] = "approved"
        flash(f"Question #{q_id} approved.", "success")
    elif action == "reject":
        df.loc[mask, "status"] = "rejected"
        flash(f"Question #{q_id} rejected.", "success")
    elif action == "edit":
        if new_q:  df.loc[mask, "question"]   = new_q
        if new_a:  df.loc[mask, "answer"]     = new_a
        if new_diff: df.loc[mask, "difficulty"] = round(max(1.0, min(5.0, new_diff)), 2)
        if new_hint: df.loc[mask, "hint"]     = new_hint
        df.loc[mask, "status"] = "approved"
        flash(f"Question #{q_id} updated and approved.", "success")

    df.to_csv(DATA_CSV, index=False)
    # Invalidate question cache
    global _cache_mtime
    _cache_mtime = 0.0

    return redirect(url_for("admin_page"))


@app.route("/admin/make-admin", methods=["POST"])
@login_required
@admin_required
def make_admin():
    """Grant admin to another user by username."""
    username = request.form.get("username", "").strip()
    user = User.query.filter_by(username=username).first()
    if not user:
        flash(f"User '{username}' not found.", "error")
    else:
        user.is_admin = True
        db.session.commit()
        flash(f"{username} is now an admin.", "success")
    return redirect(url_for("admin_page"))


if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5000)
