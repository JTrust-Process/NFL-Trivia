# 🏈 NFL Trivia — Adaptive Edition

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Gemini](https://img.shields.io/badge/Gemini-1.5%20Pro-orange.svg)](https://ai.google.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Deploy on Render](https://img.shields.io/badge/Deploy-Render-46E3B7.svg)](https://render.com/)

A full-stack adaptive NFL trivia web app. The game tracks your skill level in real time and adjusts question difficulty to match — harder questions when you're on a streak, easier ones when you struggle. Powered by a Flask backend, PostgreSQL, and Gemini AI.

🔗 **Live:** [nfl-trivia-32y5.onrender.com](https://nfl-trivia-32y5.onrender.com)

---

## ✨ Features

### Gameplay
- **Adaptive difficulty** — questions chosen based on your skill score (μ), updated after every answer using an IRT-inspired model
- **Multiple game modes** — Free Answer, Multiple Choice, Timed (15s countdown), Daily Challenge (same questions for everyone each day), and 2-player local multiplayer
- **Category + difficulty filters** — filter questions by category or difficulty tier (Easy → Legend) on the setup screen
- **Streak multipliers** — consecutive correct answers boost your skill gains
- **Hints** — use a hint for a −0.15 μ penalty

### AI Features (Google Gemini)
- **Difficulty calibration** — after enough player data accumulates, Gemini re-scores question difficulty and improves hints based on real performance
- **Personalized recommendations** — Gemini analyzes your category weaknesses and biases question selection toward your blind spots
- **Auto question generation** — when you've seen 70%+ of the question pool, Gemini silently generates 20 new questions in the background
- **Fun facts** — after a wrong answer, Gemini surfaces an interesting fact about the correct answer
- **Explanations** — after every answer, Gemini explains why the answer is correct

### Progression & Social
- **Achievement badges** — 9 unlockable achievements (First Win, On Fire, Unstoppable, Sharp, Grinder, Century, Category Master, High IQ, Legend)
- **Career μ graph** — line chart on your profile showing your skill score over time
- **Weekly leaderboard** — resets every Monday, separate from the all-time board
- **Score card** — shareable PNG image generated after each game, ready to post
- **Question submission** — players can submit questions for admin review

### Platform
- **Admin panel** — review, edit, approve, or reject AI- and player-generated questions at `/admin`
- **Password reset** — via Gmail SMTP with secure time-limited tokens
- **Mobile responsive** — optimized for phones and tablets
- **In-memory question cache** — CSV only re-read when the file changes

---

## 📂 Project Structure

```
nfl-trivia/
├── app.py                      # Flask app — all routes, game logic, AI integration
├── wsgi.py                     # Gunicorn entry point
├── render.yaml                 # Render deployment config
├── requirements.txt
├── .env                        # Local env vars (not committed)
├── data/
│   └── nfl_trivia.csv          # Question bank (id, question, answer, difficulty, category, hint, status, source)
├── src/
│   ├── trivia.py               # Skill model (IRT), question selection, answer checking
│   ├── gemini_ai.py            # Gemini — calibration, generation, personalization, facts, explanations
│   └── models.py               # SQLAlchemy models
└── templates/
    ├── index.html              # Main game UI
    ├── auth.html               # Login / register
    ├── profile.html            # Per-user stats, achievements, career graph
    ├── leaderboard.html        # All-time leaderboard
    ├── leaderboard_weekly.html # Weekly leaderboard
    ├── admin.html              # Admin question review panel
    ├── submit_question.html    # Player question submission
    ├── forgot_password.html    # Password reset request
    └── reset_password.html     # Password reset form
```

### Database Models
- `User` — accounts, email, admin flag
- `GameSession` — per-game results (μ, accuracy, streak, mode)
- `QuestionRecord` — per-question answer records used for AI calibration
- `Season` / `SeasonEntry` — weekly leaderboard tracking
- `UserAchievement` — earned achievement badges

---

## 🚀 Getting Started (Local)

### 1. Clone the repo
```bash
git clone https://github.com/JTrust-Process/NFL-Trivia.git
cd NFL-Trivia
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the project root:
```
GEMINI_API_KEY=your-gemini-api-key-here
SECRET_KEY=any-random-string-for-local-dev
MAIL_USERNAME=your-gmail@gmail.com
MAIL_PASSWORD=your-16-char-app-password
```

- Gemini API key: [aistudio.google.com](https://aistudio.google.com) (free, no credit card)
- Gmail App Password: Google Account → Security → 2-Step Verification → App Passwords

### 5. Run the app
```bash
python app.py
```
Visit `http://localhost:5000`, register an account, and start playing.

---

## 🤖 How the AI Works

### Skill Model (IRT)
Each player has a skill score μ (starts at 0.0). After every answer the score updates based on question difficulty, correctness, response time, and current streak multiplier. Questions are selected from a band near your current μ so difficulty always matches your level.

### Gemini Difficulty Calibration
After a game ends, if 30+ total answers exist with 3+ per question, Gemini reviews each question's correct rate and the average skill of players who answered it. It suggests revised difficulty scores and improved hints, written back to `nfl_trivia.csv`.

### Personalized Recommendations
At game start, your all-time category accuracy is sent to Gemini, which returns bias weights (e.g. "Draft: 2.5x, Super Bowl: 0.7x"). These weights steer `choose_next()` toward your weakest categories.

### Auto Question Generation
When you've seen 70%+ of the question pool, a background thread fires a Gemini API call to generate 20 new questions tagged as `pending` — completely silent, no delay to gameplay. They appear in the admin queue for review before going live.

### Fun Facts & Explanations
After every answer, Gemini Flash generates a short explanation of why the answer is correct. After wrong answers, it also surfaces an interesting related fact. Both are rate-limited per user to avoid API abuse.

---

## 🌐 Deploying to Render

1. Push your code to GitHub
2. Create a new **Web Service** on [render.com](https://render.com) pointed at your repo — `render.yaml` configures everything automatically
3. Create a **PostgreSQL** database named `nfl-trivia-db` in the same Render project
4. Add environment variables in the Render dashboard:
   - `GEMINI_API_KEY`
   - `MAIL_USERNAME`
   - `MAIL_PASSWORD`
5. Deploy — `SECRET_KEY` and `DATABASE_URL` are handled automatically by `render.yaml`

> **Note:** The free Render PostgreSQL database expires after 90 days. Upgrade before expiry to keep your data.

---

## 🏅 Achievements

| Badge | Icon | Condition |
|---|---|---|
| First Win | 🏆 | Complete your first game |
| On Fire | 🔥 | Get a 5-question streak |
| Unstoppable | ⚡ | Get a 10-question streak |
| Sharp | 🎯 | 90%+ accuracy in a single game |
| Grinder | 💪 | Play 10 games |
| Century | 💯 | Play 100 games |
| Category Master | 📚 | 80%+ accuracy in any category (min 10 questions) |
| High IQ | 🧠 | Reach μ 3.0 |
| Legend | 👑 | Reach μ 5.0 |

---

## 🛠️ Tech Stack

- **Backend** — Python 3.11, Flask, Flask-Login, Flask-SQLAlchemy, Flask-Mail
- **Database** — PostgreSQL (Render) / SQLite (local dev)
- **AI** — Google Gemini 1.5 Pro + Flash
- **Frontend** — Vanilla JS, HTML/CSS, Canvas API (no framework)
- **Deployment** — Render (web service + managed Postgres)