# main.py

# --- Standard library imports ---
import os
import argparse
from datetime import datetime
from pathlib import Path

# --- Third-party libraries ---
import matplotlib.pyplot as plt   # for plotting skill progression
import pandas as pd               # for storing and exporting session data

# --- Local project imports ---
from src.trivia import (
    load_questions,          # loads questions from CSV into Question objects
    update_skill,            # updates user skill based on correctness & time
    choose_next,             # picks the next question to ask
    simulate_user_response,  # generates fake answers for simulation mode
    normalize_answer,        # cleans user input for answer comparison
)

# Path to the trivia questions CSV file
DATA_CSV = "data/nfl_trivia.csv"


def run_session(mode: str, rounds: int, seed: int | None = None):
    """
    Run a trivia session, either simulated or interactive.

    Args:
        mode: "simulate" (fake answers) or "play" (user answers)
        rounds: how many questions to ask
        seed: random seed for reproducibility

    Returns:
        A pandas DataFrame with a record of the session
    """

    # Set seeds for reproducibility if provided
    if seed is not None:
        import random, numpy as np
        random.seed(seed)
        np.random.seed(seed)

    # Ensure CSV file exists
    if not Path(DATA_CSV).exists():
        raise SystemExit(f"Could not find CSV at {DATA_CSV}")

    # Load questions from CSV
    questions = load_questions(DATA_CSV)
    asked_ids = set()   # keep track of already asked questions

    mu = 0.0            # starting skill estimate
    records = []        # list to hold session data

    # Loop through the number of rounds requested
    for i in range(rounds):
        # Choose the next question based on current skill
        q = choose_next(mu, questions, asked_ids)
        if q is None:
            print("No more questions left!")
            break
        asked_ids.add(q.id)

        # --- Simulate Mode ---
        if mode == "simulate":
            outcome = simulate_user_response(mu, q.difficulty, T=8.0)
            is_correct = outcome["is_correct"]
            t = outcome["response_time"]
            p = outcome["p_correct"]
            user_ans = ""  # no actual user input in simulate mode

        # --- Play Mode (user answers in CLI) ---
        else:
            print(f"\nQ{i+1}/{rounds}  (diff {q.difficulty:.1f})  [{q.category}]")
            print(q.question)
            user_ans = input("Your answer: ").strip()
            # Compare normalized input with normalized correct answer
            is_correct = normalize_answer(user_ans) == normalize_answer(q.answer)
            t = 8.0  # placeholder until you wire real timing
            p = None

        # Update skill based on response
        mu = update_skill(mu, q.difficulty, is_correct, t)

        # Print quick feedback for this question
        print(
            f" -> correct={is_correct} | time={t:.1f}s | new_mu={mu:.2f}"
            + (f" | p_correct~{p:.2f}" if p is not None else "")
        )

        # Record the results of this question
        records.append({
            "n": i + 1,
            "question_id": q.id,
            "question": q.question,
            "answer": q.answer,
            "user_answer": user_ans,
            "category": q.category,
            "difficulty": q.difficulty,
            "is_correct": int(is_correct),
            "response_time": t,
            "mu_after": mu,
            "mode": mode,
            "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        })

    # Return as a DataFrame for plotting/saving
    return pd.DataFrame(records)


def plot_and_save(df: pd.DataFrame):
    """
    Plot skill progression and save session results to files.
    """
    if df.empty:
        print("No records to plot/save.")
        return

    os.makedirs("outputs/sessions", exist_ok=True)

    # --- Plot skill progression ---
    plt.figure()
    plt.plot(df["n"], df["mu_after"])
    plt.xlabel("Question #")
    plt.ylabel("Estimated Skill (mu)")
    plt.title("Skill Progression Over Session")
    plot_path = f"outputs/sessions/skill_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.show()

    # --- Save session records as CSV ---
    csv_path = f"outputs/sessions/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(csv_path, index=False)

    # --- Print summary stats ---
    acc = df["is_correct"].mean()
    avg_t = df["response_time"].mean()
    print("\n--- Session Summary ---")
    print(f"Questions: {len(df)}")
    print(f"Accuracy:  {acc*100:.1f}%")
    print(f"Avg time:  {avg_t:.2f}s")
    print(f"Final mu:  {df['mu_after'].iloc[-1]:.2f}")
    print(f"Saved plot -> {plot_path}")
    print(f"Saved CSV  -> {csv_path}")


def parse_args():
    """
    Parse command-line arguments for session settings.
    """
    ap = argparse.ArgumentParser(description="NFL Trivia Day 2")
    ap.add_argument("--mode", choices=["simulate", "play"], default="simulate",
                    help="simulate (no input) or play (answer in CLI)")
    ap.add_argument("--rounds", type=int, default=15, help="questions this session")
    ap.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    return ap.parse_args()


if __name__ == "__main__":
    # Parse CLI args
    args = parse_args()
    # Run session
    df = run_session(mode=args.mode, rounds=args.rounds, seed=args.seed)
    # Plot and save results
    plot_and_save(df)
