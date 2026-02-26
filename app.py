from flask import Flask, render_template, request, jsonify, session
import os, time, json, uuid
from pathlib import Path

from src.trivia import (
    load_questions, choose_next, update_skill, check_answer,
    generate_choices, SessionState, save_to_leaderboard,
    load_leaderboard, retrain_difficulty, scrape_nfl_questions,
    append_scraped_questions,
)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24))

DATA_CSV = "data/nfl_trivia.csv"
GLOBAL_HISTORY = []   # accumulates across sessions for ML retraining


def _get_questions():
    return load_questions(DATA_CSV)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/leaderboard")
def leaderboard_page():
    board = load_leaderboard()
    return render_template("leaderboard.html", board=board)


@app.route("/api/start", methods=["POST"])
def start_game():
    """Initialize a new game session."""
    data = request.json
    mode = data.get("mode", "free")           # free | multiple_choice
    timed = data.get("timed", False)          # countdown per question
    rounds = int(data.get("rounds", 15))
    multiplayer = data.get("multiplayer", False)
    player2_name = data.get("player2_name", "Player 2")
    player_name = data.get("player_name", "Anonymous")

    questions = _get_questions()
    q_pool = [{"id": q.id, "question": q.question, "answer": q.answer,
               "difficulty": q.difficulty, "category": q.category, "hint": q.hint}
              for q in questions]

    session.clear()
    session["game"] = {
        "mode": mode,
        "timed": timed,
        "rounds": rounds,
        "multiplayer": multiplayer,
        "current_round": 0,
        "asked_ids": [],
        "records": [],
        "hint_used": False,
        "q_start_time": None,
        # Player 1
        "p1_name": player_name,
        "p1_mu": 0.0,
        "p1_streak": 0,
        "p1_best_streak": 0,
        "p1_streak_multiplier": 1.0,
        "p1_cat_counts": {},
        "p1_cat_correct": {},
        # Player 2 (multiplayer only)
        "p2_name": player2_name,
        "p2_mu": 0.0,
        "p2_streak": 0,
        "p2_best_streak": 0,
        "p2_streak_multiplier": 1.0,
        "p2_cat_counts": {},
        "p2_cat_correct": {},
        "active_player": 1,
    }
    return jsonify({"status": "ok", "rounds": rounds})


@app.route("/api/question", methods=["GET"])
def get_question():
    """Return the next question for the active player."""
    g = session.get("game")
    if not g:
        return jsonify({"error": "No active game"}), 400

    if g["current_round"] >= g["rounds"]:
        return jsonify({"game_over": True})

    questions = _get_questions()
    asked = set(g["asked_ids"])

    # Build state for active player
    player = g["active_player"]
    state = _build_state(g, player)

    q = choose_next(state, questions, asked)
    if q is None:
        return jsonify({"game_over": True})

    g["asked_ids"].append(q.id)
    g["hint_used"] = False
    g["q_start_time"] = time.time()
    session["game"] = g

    choices = None
    if g["mode"] == "multiple_choice":
        choices = generate_choices(q, questions)

    return jsonify({
        "id": q.id,
        "question": q.question,
        "category": q.category,
        "difficulty": round(q.difficulty, 1),
        "choices": choices,
        "round": g["current_round"] + 1,
        "rounds": g["rounds"],
        "timed": g["timed"],
        "time_limit": 15 if g["timed"] else None,
        "multiplayer": g["multiplayer"],
        "active_player": player,
        "active_name": g[f"p{player}_name"],
        "p1_mu": round(g["p1_mu"], 2),
        "p2_mu": round(g["p2_mu"], 2) if g["multiplayer"] else None,
        "p1_streak": g["p1_streak"],
        "p2_streak": g["p2_streak"] if g["multiplayer"] else None,
    })


@app.route("/api/hint", methods=["GET"])
def get_hint():
    """Return the hint for the current question (costs 0.5 skill points)."""
    g = session.get("game")
    if not g or g.get("hint_used"):
        return jsonify({"hint": ""})

    asked = g["asked_ids"]
    if not asked:
        return jsonify({"hint": ""})

    last_id = asked[-1]
    questions = _get_questions()
    q = next((x for x in questions if x.id == last_id), None)
    hint_text = q.hint if q and q.hint and q.hint != "nan" else "No hint available for this question."

    # Penalise skill for using hint
    player = g["active_player"]
    g[f"p{player}_mu"] = round(g[f"p{player}_mu"] - 0.15, 4)
    g["hint_used"] = True
    session["game"] = g

    return jsonify({"hint": hint_text, "penalty": -0.15})


@app.route("/api/answer", methods=["POST"])
def submit_answer():
    """Process the player's answer and return feedback."""
    g = session.get("game")
    if not g:
        return jsonify({"error": "No active game"}), 400

    data = request.json
    user_ans = data.get("answer", "").strip()
    timed_out = data.get("timed_out", False)

    asked = g["asked_ids"]
    if not asked:
        return jsonify({"error": "No question active"}), 400

    last_id = asked[-1]
    questions = _get_questions()
    q = next((x for x in questions if x.id == last_id), None)
    if not q:
        return jsonify({"error": "Question not found"}), 400

    # Elapsed time
    t_elapsed = time.time() - (g["q_start_time"] or time.time())
    t_elapsed = min(t_elapsed, 60.0)

    if timed_out:
        is_correct, feedback = False, f"Time's up! The answer was: {q.answer}"
    else:
        is_correct, feedback = check_answer(user_ans, q.answer)

    player = g["active_player"]
    prev_mu = g[f"p{player}_mu"]
    multiplier = g[f"p{player}_streak_multiplier"]

    new_mu = update_skill(prev_mu, q.difficulty, is_correct, t_elapsed, multiplier)
    new_mu = round(new_mu, 4)

    # Update streak
    if is_correct:
        g[f"p{player}_streak"] += 1
        g[f"p{player}_best_streak"] = max(g[f"p{player}_best_streak"], g[f"p{player}_streak"])
    else:
        g[f"p{player}_streak"] = 0
    g[f"p{player}_streak_multiplier"] = min(2.0, 1.0 + g[f"p{player}_streak"] * 0.25)
    g[f"p{player}_mu"] = new_mu

    # Category tracking
    cat_c = g[f"p{player}_cat_counts"]
    cat_r = g[f"p{player}_cat_correct"]
    cat_c[q.category] = cat_c.get(q.category, 0) + 1
    if is_correct:
        cat_r[q.category] = cat_r.get(q.category, 0) + 1
    g[f"p{player}_cat_counts"] = cat_c
    g[f"p{player}_cat_correct"] = cat_r

    # Record for ML history
    record = {
        "question_id": q.id,
        "category": q.category,
        "difficulty": q.difficulty,
        "is_correct": int(is_correct),
        "response_time": round(t_elapsed, 2),
        "mu_before": round(prev_mu, 4),
        "mu_after": new_mu,
        "hint_used": g["hint_used"],
        "player": player,
    }
    g["records"].append(record)
    GLOBAL_HISTORY.append(record)

    # Advance round — in multiplayer, alternate players
    g["current_round"] += 1
    if g["multiplayer"]:
        g["active_player"] = 2 if player == 1 else 1

    game_over = g["current_round"] >= g["rounds"]
    session["game"] = g

    # Streak bonus message
    streak = g[f"p{player}_streak"]
    streak_msg = ""
    if streak == 3:
        streak_msg = "🔥 3 in a row!"
    elif streak == 5:
        streak_msg = "🔥🔥 5 streak! Multiplier maxing out!"
    elif streak >= 7:
        streak_msg = "🔥🔥🔥 Unstoppable!"

    return jsonify({
        "is_correct": is_correct,
        "feedback": feedback,
        "correct_answer": q.answer,
        "mu_before": round(prev_mu, 2),
        "mu_after": round(new_mu, 2),
        "delta": round(new_mu - prev_mu, 3),
        "streak": streak,
        "streak_msg": streak_msg,
        "multiplier": round(g[f"p{player}_streak_multiplier"], 2),
        "game_over": game_over,
        "active_player": player,
        "active_name": g[f"p{player}_name"],
        "p1_mu": round(g["p1_mu"], 2),
        "p2_mu": round(g["p2_mu"], 2) if g["multiplayer"] else None,
    })


@app.route("/api/results", methods=["POST"])
def save_results():
    """Save final results to leaderboard. Run ML retraining if enough data."""
    g = session.get("game")
    if not g:
        return jsonify({"error": "No active game"}), 400

    data = request.json
    p1_name = g["p1_name"]
    p1_acc = _calc_accuracy(g["records"], player=1)
    p1_score = g["p1_mu"]

    save_to_leaderboard(p1_name, p1_score, p1_acc,
                        g["rounds"], g["p1_best_streak"], g["mode"])

    p2_result = None
    if g["multiplayer"]:
        p2_acc = _calc_accuracy(g["records"], player=2)
        p2_score = g["p2_mu"]
        save_to_leaderboard(g["p2_name"], p2_score, p2_acc,
                            g["rounds"], g["p2_best_streak"], g["mode"])
        winner = g["p1_name"] if p1_score >= p2_score else g["p2_name"]
        p2_result = {"name": g["p2_name"], "mu": round(p2_score, 2),
                     "accuracy": round(p2_acc * 100, 1)}
    else:
        winner = p1_name

    # ML retraining attempt
    retrained = retrain_difficulty(DATA_CSV, GLOBAL_HISTORY)

    # Category breakdown
    cats = {}
    for cat, count in g["p1_cat_counts"].items():
        correct = g["p1_cat_correct"].get(cat, 0)
        cats[cat] = {"correct": correct, "total": count,
                     "pct": round(correct / count * 100)}

    return jsonify({
        "winner": winner,
        "p1": {"name": p1_name, "mu": round(p1_score, 2),
               "accuracy": round(p1_acc * 100, 1),
               "best_streak": g["p1_best_streak"]},
        "p2": p2_result,
        "categories": cats,
        "retrained": retrained,
        "leaderboard": load_leaderboard()[:10],
    })


@app.route("/api/scrape", methods=["POST"])
def scrape_questions():
    """Scrape new questions from Pro Football Reference."""
    new_qs = scrape_nfl_questions(max_new=10)
    added = append_scraped_questions(DATA_CSV, new_qs)
    return jsonify({"added": added, "message": f"Added {added} new questions from Pro Football Reference."})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_state(g, player):
    from collections import defaultdict
    s = SessionState()
    s.mu = g[f"p{player}_mu"]
    s.streak = g[f"p{player}_streak"]
    s.best_streak = g[f"p{player}_best_streak"]
    s.streak_multiplier = g[f"p{player}_streak_multiplier"]
    s.category_counts = defaultdict(int, g.get(f"p{player}_cat_counts", {}))
    s.category_correct = defaultdict(int, g.get(f"p{player}_cat_correct", {}))
    return s


def _calc_accuracy(records, player=1):
    player_records = [r for r in records if r.get("player") == player]
    if not player_records:
        return 0.0
    return sum(r["is_correct"] for r in player_records) / len(player_records)


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    app.run(debug=True, port=5000)