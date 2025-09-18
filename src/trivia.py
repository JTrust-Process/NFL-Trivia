from dataclasses import dataclass   # for lightweight Question class
import math                         # for sigmoid & exponential math
import random                       # for randomness in simulation/selection
import pandas as pd                 # for reading CSV files
import re                           # for cleaning user input (normalize answers)


# --- Data structure to represent each question ---
@dataclass
class Question:
    id: int
    question: str
    answer: str
    difficulty: float
    category: str


# --- Load questions from a CSV file ---
def load_questions(csv_path: str):
    """
    Load questions from CSV into a list of Question objects.
    Required columns: id, question, answer, difficulty, category
    """
    df = pd.read_csv(csv_path)   # read CSV into a DataFrame
    return [
        Question(
            id=int(row["id"]),
            question=str(row["question"]),
            answer=str(row["answer"]),
            difficulty=float(row["difficulty"]),
            category=str(row["category"])
        )
        for _, row in df.iterrows()   # iterate over rows and convert each to a Question object
    ]


# --- Sigmoid helper function ---
def sigmoid(x: float) -> float:
    """
    Sigmoid squashes input to range (0,1).
    Used for probability calculations (e.g., correctness probability).
    """
    return 1 / (1 + math.exp(-x))


# --- Update skill function ---
def update_skill(mu: float, q_difficulty: float, correct: bool, t_sec: float,
                 T: float = 8.0, alpha: float = 0.2) -> float:
    """
    Update the user's skill score (mu) based on:
      - Whether the answer was correct
      - Question difficulty vs. current skill
      - Time taken to answer (faster = stronger weight)
    
    Args:
        mu: current skill estimate
        q_difficulty: difficulty level of the question
        correct: whether the user got it right
        t_sec: time (seconds) user took to answer
        T: target time (default 8s)
        alpha: learning rate (how much to update skill each time)
    """
    # Apply time weight: slower than T → weaker update
    time_weight = math.exp(-max(0, t_sec - T) / 5)

    # Compute gradient: push mu up if correct, down if wrong
    grad = (1 - sigmoid(mu - q_difficulty)) if correct else -sigmoid(mu - q_difficulty)

    # Update skill
    return mu + alpha * time_weight * grad


# --- Choose the next question ---
def choose_next(mu: float, pool, asked_ids):
    """
    Pick the next question closest to the user's current skill level.
    Adds a bit of randomness to avoid repetition.
    """
    # Filter out already-asked questions
    candidates = [q for q in pool if q.id not in asked_ids]
    if not candidates:
        return None

    # Sort by closeness of difficulty to current skill, break ties randomly
    candidates.sort(key=lambda q: (abs(q.difficulty - mu), random.random()))

    # Pick randomly from the top 5 for variety
    return random.choice(candidates[:5])


# --- Simulate a user's response ---
def simulate_user_response(mu: float, q_difficulty: float, T: float = 8.0):
    """
    Fake user answers (used for testing/simulation).
      - Probability of correctness is higher if mu > difficulty.
      - Response time sampled from Normal(T, 2).
    """
    k = 1.2   # slope factor for probability
    p_correct = sigmoid(k * (mu - q_difficulty))   # correctness probability
    is_correct = random.random() < p_correct       # flip a biased coin
    t = random.gauss(T, 2.0)                       # sample response time ~ N(T, 2)
    t = max(2.0, min(20.0, t))                     # clamp time between 2 and 20 seconds
    return {"is_correct": is_correct, "response_time": t, "p_correct": p_correct}


# --- Normalize user answers for comparison ---
def normalize_answer(s: str) -> str:
    """
    Clean up a user's answer for fair comparison:
      - Lowercase
      - Remove punctuation
      - Collapse multiple spaces
    """
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9 ]+", "", s)  # keep only alphanumerics + spaces
    s = re.sub(r"\s+", " ", s)         # collapse multiple spaces into one
    return s
