import os, time, json
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user

from src.models import db, User, GameSession, QuestionRecord
from src.trivia import (
    load_questions, choose_next, update_skill, check_answer,
    generate_choices, SessionState, retrain_difficulty, append_scraped_questions,
)
from src.nfl_data import generate_questions_from_nfl_data

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24))

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///nfl_trivia_dev.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message = "Please log in to play."

DATA_CSV = "data/nfl_trivia.csv"


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


def init_db():
    with app.app_context():
        db.create_all()
        try:
            new_qs = generate_questions_from_nfl_data(seasons=[2022, 2023])
            added  = append_scraped_questions(DATA_CSV, new_qs)
            if added:
                print(f"[startup] Added {added} new questions from nfl-data-py")
        except Exception as e:
            print(f"[startup] nfl-data-py seed skipped: {e}")


# ── Auth ──────────────────────────────────────────────────────────────────────
@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        if not username or not password:
            flash("Username and password are required.", "error")
        elif len(username) < 3:
            flash("Username must be at least 3 characters.", "error")
        elif len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
        elif User.query.filter_by(username=username).first():
            flash("That username is taken.", "error")
        else:
            user = User(username=username)
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


@app.route("/profile")
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
    return render_template("profile.html", user=current_user,
                           sessions=sessions, cat_data=cat_data)


# ── Game API ──────────────────────────────────────────────────────────────────
@app.route("/api/start", methods=["POST"])
@login_required
def start_game():
    data = request.json
    session.clear()
    session["game"] = {
        "mode": data.get("mode", "free"),
        "timed": data.get("timed", False),
        "rounds": int(data.get("rounds", 15)),
        "multiplayer": data.get("multiplayer", False),
        "current_round": 0, "asked_ids": [], "records": [], "hint_used": False,
        "q_start_time": None,
        "p1_name": current_user.username,
        "p1_mu": 0.0, "p1_streak": 0, "p1_best_streak": 0,
        "p1_streak_multiplier": 1.0, "p1_cat_counts": {}, "p1_cat_correct": {},
        "p2_name": data.get("player2_name", "Player 2"),
        "p2_mu": 0.0, "p2_streak": 0, "p2_best_streak": 0,
        "p2_streak_multiplier": 1.0, "p2_cat_counts": {}, "p2_cat_correct": {},
        "active_player": 1,
    }
    return jsonify({"status": "ok", "rounds": session["game"]["rounds"], "p1_name": current_user.username})


@app.route("/api/question", methods=["GET"])
@login_required
def get_question():
    g = session.get("game")
    if not g:
        return jsonify({"error": "No active game"}), 400
    if g["current_round"] >= g["rounds"]:
        return jsonify({"game_over": True})
    questions = load_questions(DATA_CSV)
    player    = g["active_player"]
    state     = _build_state(g, player)
    q         = choose_next(state, questions, set(g["asked_ids"]))
    if q is None:
        return jsonify({"game_over": True})
    g["asked_ids"].append(q.id)
    g["hint_used"] = False
    g["q_start_time"] = time.time()
    session["game"] = g
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
    q = next((x for x in load_questions(DATA_CSV) if x.id == g["asked_ids"][-1]), None)
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
    q = next((x for x in load_questions(DATA_CSV) if x.id == g["asked_ids"][-1]), None)
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
    return jsonify({
        "is_correct": is_correct, "feedback": feedback, "correct_answer": q.answer,
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
    all_records = [{"question_id": r.question_id, "is_correct": r.is_correct,
                    "mu_before": r.mu_before} for r in QuestionRecord.query.all()]
    retrained = retrain_difficulty(DATA_CSV, all_records)
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
    return jsonify({"winner": winner, "retrained": retrained,
                    "p1": {"name": g["p1_name"], "mu": round(g["p1_mu"], 2),
                           "accuracy": round(p1_acc * 100, 1), "best_streak": g["p1_best_streak"]},
                    "p2": p2_result, "categories": cats, "leaderboard": leaderboard})


@app.route("/api/refresh-questions", methods=["POST"])
@login_required
def refresh_questions():
    try:
        new_qs = generate_questions_from_nfl_data(seasons=[2022, 2023, 2024])
        added  = append_scraped_questions(DATA_CSV, new_qs)
        return jsonify({"added": added, "message": f"Added {added} new questions from NFL data."})
    except Exception as e:
        return jsonify({"added": 0, "message": f"Failed: {e}"})


def _build_state(g, player):
    s = SessionState()
    s.mu = g[f"p{player}_mu"]
    s.streak = g[f"p{player}_streak"]
    s.best_streak = g[f"p{player}_best_streak"]
    s.streak_multiplier = g[f"p{player}_streak_multiplier"]
    s.category_counts  = defaultdict(int, g.get(f"p{player}_cat_counts", {}))
    s.category_correct = defaultdict(int, g.get(f"p{player}_cat_correct", {}))
    return s


if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5000)