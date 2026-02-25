# main.py

# --- Standard library imports ---
import os
import argparse
import time
from datetime import datetime
from pathlib import Path

# --- Third-party libraries ---
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np

# --- Local project imports ---
from src.trivia import (
    load_questions,
    update_skill,
    choose_next,
    simulate_user_response,
    check_answer,
    SessionState,
)

DATA_CSV = "data/nfl_trivia.csv"

# ── ANSI colour helpers (degrade gracefully on Windows) ──────────────────────
try:
    import sys
    _USE_COLOUR = sys.stdout.isatty()
except Exception:
    _USE_COLOUR = False

def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOUR else text

def green(t):  return _c(t, "32")
def red(t):    return _c(t, "31")
def yellow(t): return _c(t, "33")
def cyan(t):   return _c(t, "36")
def bold(t):   return _c(t, "1")


# ── Pretty progress bar ───────────────────────────────────────────────────────
def _progress_bar(current: int, total: int, width: int = 20) -> str:
    filled = int(width * current / total)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {current}/{total}"


# ── Session runner ────────────────────────────────────────────────────────────
def run_session(mode: str, rounds: int, seed: int | None = None) -> pd.DataFrame:
    """
    Run a trivia session (simulate or interactive play).

    Args:
        mode:   "simulate" — fake answers for testing
                "play"     — user types answers in CLI
        rounds: number of questions to ask
        seed:   optional random seed for reproducibility
    """
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)

    if not Path(DATA_CSV).exists():
        raise SystemExit(f"Could not find CSV at {DATA_CSV}")

    questions = load_questions(DATA_CSV)
    state = SessionState()   # tracks mu + per-category history
    asked_ids: set[int] = set()
    records = []

    # ── Welcome banner ──
    if mode == "play":
        print("\n" + "=" * 52)
        print(bold("  🏈  NFL TRIVIA — Adaptive Edition"))
        print("=" * 52)
        print(f"  {rounds} questions  |  difficulty adapts to you")
        print(f"  Type your best guess and press Enter")
        print("=" * 52 + "\n")

    for i in range(rounds):
        q = choose_next(state, questions, asked_ids)
        if q is None:
            print("No more questions available!")
            break
        asked_ids.add(q.id)

        # ── Simulate mode ──
        if mode == "simulate":
            outcome = simulate_user_response(state.mu, q.difficulty)
            is_correct = outcome["is_correct"]
            t = outcome["response_time"]
            p = outcome["p_correct"]
            user_ans = ""
            feedback = "✅ Correct!" if is_correct else f"❌ Wrong. Answer: {q.answer}"

        # ── Play mode ──
        else:
            # Header for this question
            bar = _progress_bar(i, rounds)
            diff_label = _difficulty_label(q.difficulty)
            print(f"  {cyan(bar)}   Skill: {state.mu:+.2f}")
            print(f"  {bold(f'Q{i+1}.')} {yellow(f'[{q.category}]')}  {diff_label}")
            print(f"  {q.question}\n")

            # Time the answer
            t_start = time.monotonic()
            try:
                user_ans = input("  Your answer: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nSession ended early.")
                break
            t = time.monotonic() - t_start

            is_correct, feedback = check_answer(user_ans, q.answer)
            p = None

            # Time badge
            time_badge = green(f"⚡ {t:.1f}s") if t < 6 else (yellow(f"🕐 {t:.1f}s") if t < 12 else red(f"🐢 {t:.1f}s"))
            print(f"\n  {green(feedback) if is_correct else red(feedback)}  {time_badge}")

        # ── Skill update ──
        prev_mu = state.mu
        state.mu = update_skill(state.mu, q.difficulty, is_correct, t)
        state.record(q.category, is_correct)

        if mode == "play":
            delta = state.mu - prev_mu
            arrow = green(f"↑ {delta:+.3f}") if delta >= 0 else red(f"↓ {delta:+.3f}")
            print(f"  Skill: {prev_mu:.2f} → {state.mu:.2f}  ({arrow})\n")
            print("  " + "─" * 48 + "\n")

        if mode == "simulate":
            print(f" -> correct={is_correct} | time={t:.1f}s | mu={state.mu:.2f}"
                  + (f" | p~{p:.2f}" if p is not None else ""))

        records.append({
            "n": i + 1,
            "question_id": q.id,
            "question": q.question,
            "answer": q.answer,
            "user_answer": user_ans,
            "category": q.category,
            "difficulty": q.difficulty,
            "is_correct": int(is_correct),
            "response_time": round(t, 2),
            "mu_after": round(state.mu, 4),
            "mode": mode,
            "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        })

    # ── End-of-session summary (play mode) ──
    if mode == "play" and records:
        df_tmp = pd.DataFrame(records)
        acc = df_tmp["is_correct"].mean()
        streak = _best_streak(df_tmp["is_correct"].tolist())
        print("\n" + "=" * 52)
        print(bold("  📊  Session Complete!"))
        print("=" * 52)
        print(f"  Questions answered : {len(df_tmp)}")
        print(f"  Accuracy           : {acc*100:.1f}%")
        print(f"  Best streak        : {streak}")
        print(f"  Final skill (μ)    : {state.mu:.2f}")
        print(f"  Avg response time  : {df_tmp['response_time'].mean():.1f}s")
        print("=" * 52 + "\n")

    return pd.DataFrame(records)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _difficulty_label(d: float) -> str:
    labels = [(1.5, "🟢 Easy"), (2.5, "🟡 Medium"), (3.5, "🟠 Hard"), (float("inf"), "🔴 Expert")]
    for threshold, label in labels:
        if d <= threshold:
            return label
    return "🔴 Expert"


def _best_streak(results: list) -> int:
    best = streak = 0
    for r in results:
        streak = streak + 1 if r else 0
        best = max(best, streak)
    return best


# ── Plot & save ───────────────────────────────────────────────────────────────
def plot_and_save(df: pd.DataFrame):
    """
    Save session CSV and render a 4-panel dashboard:
      1. Skill (μ) progression over questions
      2. Response time per question with a rolling average
      3. Accuracy by category (horizontal bar chart)
      4. Difficulty of questions asked vs. skill level
    """
    if df.empty:
        print("No records to plot/save.")
        return

    os.makedirs("outputs/sessions", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Save CSV ──
    csv_path = f"outputs/sessions/session_{ts}.csv"
    df.to_csv(csv_path, index=False)

    # ── Figure layout ──
    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("#0d1117")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    PANEL_BG   = "#161b22"
    ACCENT     = "#238636"
    LINE_COL   = "#58a6ff"
    CORR_COL   = "#3fb950"
    WRONG_COL  = "#f85149"
    DIFF_COL   = "#d29922"
    TEXT_COL   = "#c9d1d9"
    GRID_COL   = "#21262d"

    plt.rcParams.update({
        "text.color": TEXT_COL,
        "axes.labelcolor": TEXT_COL,
        "xtick.color": TEXT_COL,
        "ytick.color": TEXT_COL,
        "axes.titlecolor": TEXT_COL,
    })

    # ── Panel 1: Skill progression ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(PANEL_BG)
    ax1.spines[:].set_color(GRID_COL)
    ax1.plot(df["n"], df["mu_after"], color=LINE_COL, linewidth=2, zorder=3)
    ax1.fill_between(df["n"], df["mu_after"], alpha=0.15, color=LINE_COL)
    ax1.axhline(0, color=GRID_COL, linewidth=0.8, linestyle="--")
    # Mark correct / wrong
    correct = df[df["is_correct"] == 1]
    wrong   = df[df["is_correct"] == 0]
    ax1.scatter(correct["n"], correct["mu_after"], color=CORR_COL, s=40, zorder=4, label="Correct")
    ax1.scatter(wrong["n"],   wrong["mu_after"],   color=WRONG_COL, s=40, marker="x", zorder=4, label="Wrong")
    ax1.set_title("Skill (μ) Progression", fontweight="bold")
    ax1.set_xlabel("Question #")
    ax1.set_ylabel("μ")
    ax1.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COL)
    ax1.grid(True, color=GRID_COL, linewidth=0.5)

    # ── Panel 2: Response time ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(PANEL_BG)
    ax2.spines[:].set_color(GRID_COL)
    colors = [CORR_COL if c else WRONG_COL for c in df["is_correct"]]
    ax2.bar(df["n"], df["response_time"], color=colors, alpha=0.8)
    if len(df) >= 3:
        roll = df["response_time"].rolling(3, center=True).mean()
        ax2.plot(df["n"], roll, color="white", linewidth=1.5, linestyle="--", label="3-Q avg")
        ax2.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COL)
    ax2.set_title("Response Time per Question", fontweight="bold")
    ax2.set_xlabel("Question #")
    ax2.set_ylabel("Seconds")
    ax2.grid(True, axis="y", color=GRID_COL, linewidth=0.5)

    # ── Panel 3: Accuracy by category ──
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(PANEL_BG)
    ax3.spines[:].set_color(GRID_COL)
    cat_stats = df.groupby("category")["is_correct"].agg(["mean", "count"]).reset_index()
    cat_stats = cat_stats.sort_values("mean")
    bar_colors = [CORR_COL if v >= 0.5 else WRONG_COL for v in cat_stats["mean"]]
    bars = ax3.barh(cat_stats["category"], cat_stats["mean"] * 100, color=bar_colors, alpha=0.85)
    # Annotate with count
    for bar, (_, row) in zip(bars, cat_stats.iterrows()):
        ax3.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 f"n={int(row['count'])}", va="center", fontsize=7.5, color=TEXT_COL)
    ax3.axvline(50, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
    ax3.set_title("Accuracy by Category (%)", fontweight="bold")
    ax3.set_xlabel("Accuracy (%)")
    ax3.set_xlim(0, 115)
    ax3.grid(True, axis="x", color=GRID_COL, linewidth=0.5)

    # ── Panel 4: Difficulty asked vs. skill ──
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(PANEL_BG)
    ax4.spines[:].set_color(GRID_COL)
    ax4.plot(df["n"], df["difficulty"], color=DIFF_COL, linewidth=1.5, label="Q difficulty", zorder=2)
    ax4.plot(df["n"], df["mu_after"],   color=LINE_COL,  linewidth=1.5, linestyle="--", label="Skill μ",  zorder=2)
    ax4.fill_between(df["n"], df["difficulty"], df["mu_after"],
                     where=df["difficulty"] >= df["mu_after"],
                     alpha=0.1, color=WRONG_COL, label="Above skill")
    ax4.fill_between(df["n"], df["difficulty"], df["mu_after"],
                     where=df["difficulty"] < df["mu_after"],
                     alpha=0.1, color=CORR_COL, label="Below skill")
    ax4.set_title("Difficulty vs. Skill Over Session", fontweight="bold")
    ax4.set_xlabel("Question #")
    ax4.set_ylabel("Level")
    ax4.legend(fontsize=7.5, facecolor=PANEL_BG, edgecolor=GRID_COL)
    ax4.grid(True, color=GRID_COL, linewidth=0.5)

    # ── Super title ──
    acc = df["is_correct"].mean()
    fig.suptitle(
        f"NFL Trivia Session  |  {len(df)} questions  |  Accuracy {acc*100:.1f}%  |  Final μ = {df['mu_after'].iloc[-1]:.2f}",
        fontsize=11, fontweight="bold", color=TEXT_COL, y=0.98,
    )

    plot_path = f"outputs/sessions/skill_{ts}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()

    print(f"  Saved chart → {plot_path}")
    print(f"  Saved CSV   → {csv_path}")


# ── CLI argument parsing ──────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser(
        description="NFL Trivia — Adaptive Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                        # simulate 15 questions
  python main.py --mode play            # interactive game, 15 questions
  python main.py --mode play --rounds 20
  python main.py --mode simulate --seed 42
        """,
    )
    ap.add_argument("--mode",   choices=["simulate", "play"], default="simulate",
                    help="simulate (auto-answers) or play (you answer)")
    ap.add_argument("--rounds", type=int, default=15,
                    help="number of questions per session (default: 15)")
    ap.add_argument("--seed",   type=int, default=None,
                    help="random seed for reproducibility")
    ap.add_argument("--no-plot", action="store_true",
                    help="skip the chart and only save CSV")
    return ap.parse_args()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    df = run_session(mode=args.mode, rounds=args.rounds, seed=args.seed)
    if not args.no_plot:
        plot_and_save(df)