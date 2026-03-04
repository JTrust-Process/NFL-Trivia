from dataclasses import dataclass, field
import math, random, re
from collections import defaultdict

import pandas as pd


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


def choose_next(state, pool, asked_ids, personalized_weights=None):
    """
    Pick the next question adaptively.
    personalized_weights: optional dict of {category: float} from Gemini,
    where higher weights mean the player needs more practice in that category.
    """
    candidates = [q for q in pool if q.id not in asked_ids]
    if not candidates:
        return None

    band = 1.5
    while band <= 10:
        in_band = [q for q in candidates if abs(q.difficulty - state.mu) <= band]
        if len(in_band) >= 5:
            break
        band += 0.5
    if not in_band:
        in_band = candidates

    def score(q):
        diff_penalty = abs(q.difficulty - state.mu)
        base_score = 1.0 / (1 + diff_penalty)
        variety_bonus = 0.4 / (1 + state.category_counts[q.category])
        personal_mult = 1.0
        if personalized_weights:
            personal_mult = personalized_weights.get(q.category, 1.0)
        return max(0.01, (base_score + variety_bonus) * personal_mult)

    scores = [score(q) for q in in_band]
    top_n = min(6, len(in_band))
    paired = sorted(zip(scores, in_band), key=lambda x: x[0], reverse=True)[:top_n]
    top_scores = [s for s, _ in paired]
    top_qs = [q for _, q in paired]
    total = sum(top_scores)
    weights = [s / total for s in top_scores]
    return random.choices(top_qs, weights=weights, k=1)[0]


def generate_choices(correct, pool, n_wrong=3):
    same_cat = [q for q in pool if q.id != correct.id and q.category == correct.category]
    other    = [q for q in pool if q.id != correct.id and q.category != correct.category]
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


def _extract_number(s):
    m = re.search(r"\d+", s)
    return int(m.group()) if m else None


def check_answer(user_ans, correct_ans):
    u = normalize_answer(user_ans)
    c = normalize_answer(correct_ans)
    if u == c:
        return True, "Correct!"
    if not u:
        return False, f"Wrong. The answer was: {correct_ans}"
    c_num = _extract_number(c)
    u_num = _extract_number(u)
    if c_num is not None and u_num is not None:
        if c_num == u_num:
            return True, "Correct!"
        if c.strip().isdigit() and abs(c_num - u_num) <= 1:
            return True, "Close — accepted!"
        return False, f"Wrong. The answer was: {correct_ans}"
    u_words = set(u.split())
    c_words = set(c.split())
    _generic = {"new", "los", "san", "city", "bay", "north", "south", "east",
                "west", "york", "england", "angeles", "francisco", "kansas",
                "green", "tampa", "carolina", "dallas", "denver", "miami",
                "atlanta", "detroit", "houston", "indiana", "chicago"}
    distinctive_c = {w for w in c_words if len(w) >= 4 and w not in _generic}
    if u_words & distinctive_c:
        return True, "Close enough — accepted!"
    if len(u_words) >= 2:
        overlap = len(u_words & c_words)
        if overlap >= 2 or (len(c_words) > 0 and overlap / len(c_words) >= 0.5):
            return True, "Close enough — accepted!"
    if len(u) >= 5 and len(u) >= len(c) * 0.5:
        if u in c or c in u:
            return True, "Close enough — accepted!"
    return False, f"Wrong. The answer was: {correct_ans}"