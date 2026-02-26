from dataclasses import dataclass, field
import math, random, re, json, os
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np


@dataclass
class Question:
    id: int
    question: str
    answer: str
    difficulty: float
    category: str
    hint: str = ""


@dataclass
class SessionState:
    mu: float = 0.0
    streak: int = 0
    best_streak: int = 0
    streak_multiplier: float = 1.0
    category_counts: dict = field(default_factory=lambda: defaultdict(int))
    category_correct: dict = field(default_factory=lambda: defaultdict(int))

    def accuracy_for(self, cat):
        n = self.category_counts[cat]
        return (self.category_correct[cat] / n) if n else 0.5

    def record(self, cat, correct):
        self.category_counts[cat] += 1
        if correct:
            self.category_correct[cat] += 1
            self.streak += 1
            self.best_streak = max(self.best_streak, self.streak)
        else:
            self.streak = 0
        self.streak_multiplier = min(2.0, 1.0 + self.streak * 0.25)


def load_questions(csv_path):
    df = pd.read_csv(csv_path)
    if "hint" not in df.columns:
        df["hint"] = ""
    return [
        Question(
            id=int(row["id"]),
            question=str(row["question"]),
            answer=str(row["answer"]),
            difficulty=float(row["difficulty"]),
            category=str(row["category"]),
            hint=str(row.get("hint", "") or ""),
        )
        for _, row in df.iterrows()
    ]


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def update_skill(mu, q_diff, correct, t_sec, multiplier=1.0, T=8.0, alpha=0.25):
    time_weight = math.exp(-max(0.0, t_sec - T) / 6.0)
    expected = sigmoid(mu - q_diff)
    delta = (1.0 - expected) if correct else -expected
    return mu + alpha * time_weight * delta * multiplier


def choose_next(state, pool, asked_ids):
    candidates = [q for q in pool if q.id not in asked_ids]
    if not candidates:
        return None
    band = 1.5
    while band <= 10:
        in_band = [q for q in candidates if abs(q.difficulty - state.mu) <= band]
        if len(in_band) >= 3:
            break
        band += 0.5
    if not in_band:
        in_band = candidates

    def score(q):
        return -abs(q.difficulty - state.mu) + 0.3 / (1 + state.category_counts[q.category]) + random.uniform(0, 0.15)

    in_band.sort(key=score, reverse=True)
    return in_band[0]


def generate_choices(correct, pool, n_wrong=3):
    same_cat = [q for q in pool if q.id != correct.id and q.category == correct.category]
    other = [q for q in pool if q.id != correct.id and q.category != correct.category]
    same_cat.sort(key=lambda q: abs(q.difficulty - correct.difficulty))
    other.sort(key=lambda q: abs(q.difficulty - correct.difficulty))
    distractors_pool = same_cat + other
    seen = {normalize_answer(correct.answer)}
    distractors = []
    for q in distractors_pool:
        norm = normalize_answer(q.answer)
        if norm not in seen:
            distractors.append(q.answer)
            seen.add(norm)
        if len(distractors) == n_wrong:
            break
    fallbacks = ["Pittsburgh Steelers", "San Francisco 49ers", "New York Giants",
                 "Chicago Bears", "Tom Brady", "Jerry Rice", "1967", "2004"]
    for fb in fallbacks:
        if len(distractors) >= n_wrong:
            break
        if normalize_answer(fb) not in seen:
            distractors.append(fb)
            seen.add(normalize_answer(fb))
    choices = distractors[:n_wrong] + [correct.answer]
    random.shuffle(choices)
    return choices


def normalize_answer(s):
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    s = re.sub(r"\b(the|a|an)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def check_answer(user_ans, correct_ans):
    u = normalize_answer(user_ans)
    c = normalize_answer(correct_ans)
    if u == c:
        return True, "Correct!"
    if u and (u in c or c in u):
        return True, "Close enough — accepted!"
    return False, f"Wrong. The answer was: {correct_ans}"


LEADERBOARD_PATH = os.environ.get("LEADERBOARD_PATH", "outputs/leaderboard.json")

def load_leaderboard():
    if Path(LEADERBOARD_PATH).exists():
        with open(LEADERBOARD_PATH) as f:
            return json.load(f)
    return []


def save_to_leaderboard(name, score, accuracy, rounds, best_streak, mode):
    os.makedirs("outputs", exist_ok=True)
    board = load_leaderboard()
    board.append({
        "name": name,
        "score": round(score, 3),
        "accuracy": round(accuracy * 100, 1),
        "rounds": rounds,
        "best_streak": best_streak,
        "mode": mode,
    })
    board.sort(key=lambda x: x["score"], reverse=True)
    board = board[:20]
    with open(LEADERBOARD_PATH, "w") as f:
        json.dump(board, f, indent=2)
    return board


def retrain_difficulty(csv_path, session_history):
    if len(session_history) < 30:
        return False
    try:
        df_hist = pd.DataFrame(session_history)
        q_stats = df_hist.groupby("question_id").agg(
            correct_rate=("is_correct", "mean"),
            avg_mu=("mu_before", "mean"),
            count=("is_correct", "count"),
        ).reset_index()
        q_stats = q_stats[q_stats["count"] >= 3]
        if q_stats.empty:
            return False

        def safe_logit(p):
            p = max(0.05, min(0.95, p))
            return math.log(p / (1 - p))

        q_stats["new_difficulty"] = q_stats.apply(
            lambda r: r["avg_mu"] - safe_logit(r["correct_rate"]) * 0.8, axis=1
        ).clip(1, 5).round(2)

        df_csv = pd.read_csv(csv_path)
        updates = dict(zip(q_stats["question_id"], q_stats["new_difficulty"]))
        df_csv["difficulty"] = df_csv.apply(
            lambda r: updates.get(r["id"], r["difficulty"]), axis=1
        )
        df_csv.to_csv(csv_path, index=False)
        return True
    except Exception as e:
        print(f"ML retrain skipped: {e}")
        return False


def scrape_nfl_questions(max_new=10):
    try:
        import requests
        from bs4 import BeautifulSoup
        headers = {"User-Agent": "Mozilla/5.0 (NFL Trivia personal project)"}
        url = "https://www.pro-football-reference.com/leaders/pass_td_single_season.htm"
        r = requests.get(url, headers=headers, timeout=8)
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        table = soup.find("table", {"id": "leaders"})
        if not table:
            return []
        questions = []
        rows = table.find("tbody").find_all("tr")[:max_new]
        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 4:
                continue
            rank = row.find("th").text.strip() if row.find("th") else "?"
            player = cols[0].text.strip()
            year = cols[1].text.strip()
            tds = cols[3].text.strip()
            if not (player and year and tds):
                continue
            questions.append({
                "question": f"How many touchdown passes did {player} throw in {year}?",
                "answer": tds,
                "difficulty": 4,
                "category": "Records",
                "hint": f"It was one of the top single-season TD records (rank #{rank})",
            })
        return questions
    except Exception as e:
        print(f"Scraping skipped: {e}")
        return []


def append_scraped_questions(csv_path, new_qs):
    if not new_qs:
        return 0
    df = pd.read_csv(csv_path)
    existing = set(df["question"].str.lower())
    max_id = df["id"].max()
    added = 0
    for q in new_qs:
        if q["question"].lower() not in existing:
            q["id"] = max_id + 1
            max_id += 1
            df = pd.concat([df, pd.DataFrame([q])], ignore_index=True)
            existing.add(q["question"].lower())
            added += 1
    if added:
        df.to_csv(csv_path, index=False)
    return added