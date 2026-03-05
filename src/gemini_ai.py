"""
src/gemini_ai.py
────────────────
Three Gemini-powered features:

1. smarter_difficulty_calibration()
   Sends player performance data to Gemini and gets back refined
   difficulty scores + improved hints for questions.

2. auto_generate_questions()
   Called when a player has seen 70%+ of the question pool.
   Silently generates 20 new questions and appends to the CSV.

3. get_personalized_weights()
   Returns per-category bias weights based on a player's weak spots,
   used by choose_next() to steer question selection.
"""

import os
import time
import re
import json
import threading
import pandas as pd

GEMINI_MODEL = "gemini-2.0-flash"   # used for all calls (fast + capable)
DATA_CSV     = "data/nfl_trivia.csv"

# ── Gemini client (lazy init) ─────────────────────────────────────────────────
_client = None

def _get_client():
    global _client
    if _client is None:
        try:
            from google import genai
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY environment variable not set")
            _client = genai.Client(api_key=api_key)
        except ImportError:
            raise RuntimeError("google-genai not installed — run: pip install google-genai")
    return _client


def _call_gemini(prompt: str) -> str:
    """Make a Gemini API call and return raw text response."""
    client = _get_client()
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return response.text.strip()


def _parse_json(raw: str) -> any:
    """Strip markdown fences and parse JSON safely."""
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


# ── Feature 1: Smarter Difficulty Calibration ────────────────────────────────
def smarter_difficulty_calibration(csv_path: str, session_history: list) -> bool:
    """
    Uses Gemini to reason about question difficulty based on real player data.
    More nuanced than the pure logit formula — considers question wording,
    topic obscurity, and answer ambiguity alongside raw correct rates.

    Only runs when there are 3+ answers per question to avoid noise.
    Returns True if CSV was updated.
    """
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

        # Load question text for context
        df_csv = pd.read_csv(csv_path)
        q_lookup = df_csv.set_index("id")[["question", "answer", "difficulty", "hint"]].to_dict("index")

        # Build context for Gemini — only send questions with enough data
        questions_data = []
        for _, row in q_stats.iterrows():
            qid = int(row["question_id"])
            if qid not in q_lookup:
                continue
            q = q_lookup[qid]
            questions_data.append({
                "id": qid,
                "question": q["question"],
                "answer": q["answer"],
                "current_difficulty": round(float(q["difficulty"]), 2),
                "correct_rate": round(float(row["correct_rate"]), 3),
                "avg_player_skill": round(float(row["avg_mu"]), 3),
                "times_answered": int(row["count"]),
                "current_hint": q.get("hint", ""),
            })

        if not questions_data:
            return False

        prompt = f"""You are an NFL trivia difficulty calibration system.

I have player performance data for {len(questions_data)} trivia questions.
For each question, analyze:
- The question wording and topic (is it obscure? ambiguous? well-known?)
- The correct_rate (how often players got it right)
- The avg_player_skill (μ score of players who attempted it, higher = better players)
- Whether the current_difficulty seems accurate

Difficulty scale: 1.0 = easiest, 5.0 = hardest

Rules:
- If correct_rate > 0.80 and avg_player_skill < 1.0, question is probably too easy → lower difficulty
- If correct_rate < 0.30 and avg_player_skill > 0.5, question is probably too hard → raise difficulty  
- If the question is about a very obscure stat or player, keep difficulty high even with decent correct rate
- Also suggest an improved hint if the current one is weak or missing

Return ONLY a JSON array. Each element:
{{"id": <int>, "new_difficulty": <float 1.0-5.0>, "improved_hint": "<string or null if hint is already good>"}}

Questions data:
{json.dumps(questions_data, indent=2)}"""

        raw = _call_gemini(prompt)
        updates = _parse_json(raw)

        if not isinstance(updates, list):
            print("Gemini calibration: unexpected response format")
            return False

        # Apply updates to CSV
        update_map = {u["id"]: u for u in updates if "id" in u and "new_difficulty" in u}
        changed = 0
        for idx, row in df_csv.iterrows():
            qid = int(row["id"])
            if qid in update_map:
                u = update_map[qid]
                new_diff = max(1.0, min(5.0, float(u["new_difficulty"])))
                df_csv.at[idx, "difficulty"] = round(new_diff, 2)
                if u.get("improved_hint") and str(u["improved_hint"]).strip():
                    df_csv.at[idx, "hint"] = str(u["improved_hint"]).strip()
                changed += 1

        if changed:
            df_csv.to_csv(csv_path, index=False)
            print(f"[Gemini] Calibrated difficulty for {changed} questions")
            return True

        return False

    except Exception as e:
        print(f"[Gemini] Difficulty calibration skipped: {e}")
        return False


# ── Feature 2: Auto-Generate Questions When Pool Runs Low ────────────────────
def auto_generate_questions(csv_path: str, asked_ids: list, user_id: int) -> None:
    """
    Checks if the player has seen 70%+ of the question pool.
    If so, generates 20 new questions in a background thread.
    Fire-and-forget — doesn't block the game response.
    """
    try:
        df = pd.read_csv(csv_path)
        total = len(df)
        seen_pct = len(asked_ids) / total if total > 0 else 0

        if seen_pct < 0.70:
            return  # Pool is fine, no action needed

        print(f"[Gemini] User {user_id} has seen {seen_pct:.0%} of questions — generating more...")

        # Fire background thread so the game response isn't delayed
        thread = threading.Thread(
            target=_generate_and_append,
            args=(csv_path, df),
            daemon=True
        )
        thread.start()

    except Exception as e:
        print(f"[Gemini] Auto-generate check failed: {e}")


def _generate_and_append(csv_path: str, existing_df: pd.DataFrame) -> None:
    """Background worker: generates 20 new questions and appends to CSV."""
    try:
        existing_questions = set(existing_df["question"].str.strip().str.lower())
        categories = existing_df["category"].value_counts().to_dict()

        # Weight generation toward categories that already have the most questions
        # (keeps the distribution balanced)
        cat_list = list(categories.keys())

        prompt = f"""You are an NFL trivia question writer. Generate exactly 20 new NFL trivia questions.

Distribute them across these categories (generate more for categories with more existing questions):
{json.dumps(categories, indent=2)}

Rules:
- Questions must be factually accurate NFL knowledge
- Vary difficulty: mix of easy (1-2), medium (3), hard (4-5)
- Vary formats: "Who...", "Which team...", "What year...", "How many...", "In what city..."
- Answers should be concise (1-5 words)
- Do NOT duplicate these existing question topics (sample):
{json.dumps(list(existing_questions)[:30], indent=2)}

Return ONLY a JSON array. Each element must have exactly these keys:
{{"question": "...", "answer": "...", "difficulty": <float 1-5>, "category": "...", "hint": "..."}}"""

        raw = _call_gemini(prompt)
        new_qs = _parse_json(raw)

        if not isinstance(new_qs, list):
            print("[Gemini] Auto-generate: unexpected response format")
            return

        # Re-read CSV fresh in case it changed since we started
        df = pd.read_csv(csv_path)
        existing_lower = set(df["question"].str.strip().str.lower())
        max_id = int(df["id"].max())
        added = 0

        for q in new_qs:
            if not all(k in q for k in ("question", "answer", "difficulty", "category", "hint")):
                continue
            if q["question"].strip().lower() in existing_lower:
                continue
            max_id += 1
            new_row = {
                "id": max_id,
                "question": str(q["question"]).strip(),
                "answer": str(q["answer"]).strip(),
                "difficulty": round(max(1.0, min(5.0, float(q["difficulty"]))), 2),
                "category": str(q["category"]).strip(),
                "hint": str(q["hint"]).strip(),
                "status": "pending",   # requires admin review before going live
                "source": "gemini",
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            existing_lower.add(q["question"].strip().lower())
            added += 1

        if added:
            df.to_csv(csv_path, index=False)
            print(f"[Gemini] Auto-generated {added} new questions")

    except Exception as e:
        print(f"[Gemini] Auto-generate background task failed: {e}")


# ── Feature 3: Personalized Category Weights ─────────────────────────────────
def get_personalized_weights(cat_counts: dict, cat_correct: dict) -> dict:
    """
    Analyzes a player's category performance and returns bias weights
    for choose_next() to steer toward weak spots.

    Returns a dict of {category: weight_multiplier} where:
    - weak categories (low accuracy) get higher weights → asked more often
    - strong categories get lower weights → asked less often
    - categories never seen get neutral weight

    Uses Gemini to reason about which gaps matter most for NFL knowledge,
    not just raw accuracy numbers.
    """
    if not cat_counts or sum(cat_counts.values()) < 10:
        return {}  # Not enough data yet, use default behavior

    try:
        # Build accuracy profile
        profile = {}
        for cat, total in cat_counts.items():
            correct = cat_correct.get(cat, 0)
            profile[cat] = {
                "total_answered": total,
                "correct": correct,
                "accuracy": round(correct / total, 3) if total > 0 else 0.0,
            }

        prompt = f"""You are an NFL trivia coaching system analyzing a player's knowledge gaps.

Here is the player's performance by category:
{json.dumps(profile, indent=2)}

Your job: return bias weights so the game asks more questions in weak areas.

Rules:
- Weight range: 0.5 (strong, ask less) to 3.0 (very weak, ask much more)
- Neutral weight is 1.0
- Categories with < 5 answers should stay near 1.0 (not enough data)
- Categories with accuracy < 0.40 should get weight 2.0-3.0
- Categories with accuracy > 0.75 should get weight 0.5-0.8
- Consider which knowledge gaps are most important for well-rounded NFL knowledge

Return ONLY a JSON object mapping category name to weight float.
Example: {{"Super Bowl": 1.2, "Players": 2.5, "Records": 0.7}}

Player profile to analyze:
{json.dumps(profile, indent=2)}"""

        raw = _call_gemini(prompt)
        weights = _parse_json(raw)

        if not isinstance(weights, dict):
            return {}

        # Validate all weights are reasonable floats
        clean = {}
        for cat, w in weights.items():
            try:
                clean[cat] = max(0.3, min(3.0, float(w)))
            except (TypeError, ValueError):
                clean[cat] = 1.0

        return clean

    except Exception as e:
        print(f"[Gemini] Personalized weights skipped: {e}")
        return {}


# ── Feature 4: Fun Fact After Wrong Answer ────────────────────────────────────

# Simple in-process rate limiter — tracks last call time per user
# to avoid hammering Gemini on every wrong answer
_fact_last_called: dict = {}   # user_id -> timestamp
_FACT_COOLDOWN = 20            # seconds between fun fact calls per user

# Flash-speed model for live gameplay (faster than Pro)
_FLASH_MODEL = "gemini-2.0-flash"
_flash_client = None

def _get_flash_client():
    global _flash_client
    if _flash_client is None:
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        _flash_client = genai.Client(api_key=api_key)
    return _flash_client


def get_fun_fact(question: str, correct_answer: str, category: str, user_id: int) -> str:
    """
    Returns a single interesting fact about the correct answer to show
    after a player gets a question wrong.

    Rate limited to once every 20 seconds per user to avoid API abuse.
    Returns an empty string if rate limited or on any error, so gameplay
    is never blocked.
    """
    now = time.time()
    last = _fact_last_called.get(user_id, 0)
    if now - last < _FACT_COOLDOWN:
        return ""  # Rate limited — silently skip

    _fact_last_called[user_id] = now

    try:
        prompt = (
            f"NFL trivia question: {question}\n"
            f"Correct answer: {correct_answer}\n"
            f"Category: {category}\n\n"
            f"Write ONE interesting fact about '{correct_answer}' related to this question. "
            f"Keep it to a single sentence, max 20 words. "
            f"Be specific and surprising — not just a restatement of the answer. "
            f"Do not start with 'Did you know'. No emojis. Plain text only."
        )
        client = _get_flash_client()
        response = client.models.generate_content(model=_FLASH_MODEL, contents=prompt)
        fact = response.text.strip().strip('"').strip("'")
        # Sanity check — reject if too long or empty
        if not fact or len(fact) > 200:
            return ""
        return fact
    except Exception as e:
        print(f"[Gemini] Fun fact skipped: {e}")
        return ""


# ── Feature 5: Explanation After Every Answer ─────────────────────────────────

_explanation_last_called: dict = {}
_EXPLANATION_COOLDOWN = 15  # seconds between explanation calls per user

def get_explanation(question: str, correct_answer: str, category: str,
                    is_correct: bool, user_id: int) -> str:
    """
    Returns a short explanation of why the answer is correct.
    Shown after every answer — correct or wrong.
    Rate limited to once every 15 seconds per user.
    Returns empty string if rate limited or on error.
    """
    now = time.time()
    last = _explanation_last_called.get(f"exp_{user_id}", 0)
    if now - last < _EXPLANATION_COOLDOWN:
        return ""

    _explanation_last_called[f"exp_{user_id}"] = now

    try:
        tone = "reinforce why this is correct" if is_correct else "explain why the correct answer is what it is"
        prompt = (
            f"NFL trivia question: {question}\n"
            f"Correct answer: {correct_answer}\n"
            f"Category: {category}\n\n"
            f"In 1-2 sentences, {tone}. "
            f"Be specific, educational, and concise. "
            f"Do not restate the question. No emojis. Plain text only."
        )
        client = _get_flash_client()
        response = client.models.generate_content(model=_FLASH_MODEL, contents=prompt)
        explanation = response.text.strip().strip('"').strip("'")
        if not explanation or len(explanation) > 300:
            return ""
        return explanation
    except Exception as e:
        print(f"[Gemini] Explanation skipped: {e}")
        return ""