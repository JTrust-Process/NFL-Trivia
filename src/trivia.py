from dataclasses import dataclass, field   # for lightweight Question class
import math                                # for sigmoid & exponential math
import random                              # for randomness in simulation/selection
import pandas as pd                        # for reading CSV files
import re                                  # for cleaning user input (normalize answers)
from collections import defaultdict        # for category tracking


# --- Data structure to represent each question ---
@dataclass
class Question:
    id: int
    question: str
    answer: str
    difficulty: float
    category: str


# --- Session state for adaptive logic ---
@dataclass
class SessionState:
    """
    Tracks per-category performance alongside overall skill (mu).
    Used to ensure category variety and smarter question targeting.
    """
    mu: float = 0.0
    category_counts: dict = field(default_factory=lambda: defaultdict(int))
    category_correct: dict = field(default_factory=lambda: defaultdict(int))

    def accuracy_for(self, category: str) -> float:
        count = self.category_counts[category]
        if count == 0:
            return 0.5   # neutral prior for unseen categories
        return self.category_correct[category] / count

    def record(self, category: str, is_correct: bool):
        self.category_counts[category] += 1
        if is_correct:
            self.category_correct[category] += 1


# --- Load questions from a CSV file ---
def load_questions(csv_path: str) -> list[Question]:
    """
    Load questions from CSV into a list of Question objects.
    Required columns: id, question, answer, difficulty, category
    """
    df = pd.read_csv(csv_path)
    return [
        Question(
            id=int(row["id"]),
            question=str(row["question"]),
            answer=str(row["answer"]),
            difficulty=float(row["difficulty"]),
            category=str(row["category"])
        )
        for _, row in df.iterrows()
    ]


# --- Sigmoid helper function ---
def sigmoid(x: float) -> float:
    """Squashes input to (0, 1). Used for probability estimates."""
    return 1 / (1 + math.exp(-x))


# --- Update skill function ---
def update_skill(mu: float, q_difficulty: float, correct: bool, t_sec: float,
                 T: float = 8.0, alpha: float = 0.25) -> float:
    """
    IRT-inspired skill update (Item Response Theory).

    Updates mu based on:
      - Correctness vs. expected correctness given current mu
      - Time weight: answering faster than target boosts the update slightly
      - Learning rate alpha controls how fast mu moves

    The update is larger when the result is surprising:
      - Getting a hard question right pushes mu up more
      - Getting an easy question wrong drops mu more
    """
    time_weight = math.exp(-max(0.0, t_sec - T) / 6.0)   # decay for slow answers
    expected = sigmoid(mu - q_difficulty)                 # probability we expected user to get right
    delta = (1.0 - expected) if correct else -expected    # surprise factor
    return mu + alpha * time_weight * delta


# --- Choose the next question (adaptive) ---
def choose_next(state: SessionState, pool: list[Question], asked_ids: set,
                window: float = 1.5) -> Question | None:
    """
    Adaptive question selection using skill level + category balancing.

    Strategy:
      1. Filter questions within a difficulty window around current mu.
         If no candidates found, widen the window until we find some.
      2. Among those, prefer categories the user has seen least (variety boost).
      3. Add small random noise to keep things unpredictable.

    Args:
        state: current session state with mu and category history
        pool: all available questions
        asked_ids: set of already-asked question IDs
        window: initial difficulty band around mu to search within
    """
    candidates = [q for q in pool if q.id not in asked_ids]
    if not candidates:
        return None

    # Expand window until we have at least 3 candidates to choose from
    band = window
    while band <= 10:
        in_band = [q for q in candidates if abs(q.difficulty - state.mu) <= band]
        if len(in_band) >= 3:
            break
        band += 0.5
    if not in_band:
        in_band = candidates   # fallback: use everything

    # Score each candidate: reward closeness to mu + penalize over-asked categories
    def score(q: Question) -> float:
        difficulty_score = -abs(q.difficulty - state.mu)
        # Slight bonus for categories seen less often (encourages variety)
        seen = state.category_counts[q.category]
        variety_bonus = 0.3 / (1 + seen)
        noise = random.uniform(0, 0.15)   # small noise so it's not fully deterministic
        return difficulty_score + variety_bonus + noise

    in_band.sort(key=score, reverse=True)
    return in_band[0]   # return the top-scored candidate


# --- Simulate a user's response ---
def simulate_user_response(mu: float, q_difficulty: float, T: float = 8.0) -> dict:
    """
    Generate fake user answers for testing/simulation mode.
    Correctness probability is based on IRT sigmoid of (mu - difficulty).
    Response time sampled from N(T, 2) and clamped to [2, 20].
    """
    k = 1.2
    p_correct = sigmoid(k * (mu - q_difficulty))
    is_correct = random.random() < p_correct
    t = max(2.0, min(20.0, random.gauss(T, 2.0)))
    return {"is_correct": is_correct, "response_time": t, "p_correct": p_correct}


# --- Normalize user answers for comparison ---
def normalize_answer(s: str) -> str:
    """
    Clean and normalize an answer string for fair comparison.
    Strips punctuation, lowercases, collapses whitespace.
    Also removes common filler words like 'the', 'a', 'an'.
    """
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    s = re.sub(r"\b(the|a|an)\b", "", s)   # remove articles
    s = re.sub(r"\s+", " ", s).strip()
    return s


# --- Check answer with partial credit for close matches ---
def check_answer(user_ans: str, correct_ans: str) -> tuple[bool, str]:
    """
    Compare user answer to correct answer.
    Returns (is_correct, feedback_message).

    Handles:
      - Exact match after normalization
      - Partial match: user answer is a substring of correct or vice versa
        (e.g. "Brady" matching "Tom Brady")
    """
    user_norm = normalize_answer(user_ans)
    correct_norm = normalize_answer(correct_ans)

    if user_norm == correct_norm:
        return True, "✅ Correct!"

    # Partial match: one contains the other (last name, nickname, etc.)
    if user_norm and (user_norm in correct_norm or correct_norm in user_norm):
        return True, "✅ Close enough — accepted!"

    return False, f"❌ Wrong. The answer was: {correct_ans}"