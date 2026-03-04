# 🏈 NFL Trivia — Adaptive Edition

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Gemini](https://img.shields.io/badge/Gemini-1.5%20Pro-orange.svg)](https://ai.google.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Deploy on Render](https://img.shields.io/badge/Deploy-Render-46E3B7.svg)](https://render.com/)

A full-stack adaptive NFL trivia web app. The game tracks your skill level in real time and adjusts question difficulty to match — harder questions when you're on a streak, easier ones when you struggle. Powered by a Flask backend, PostgreSQL, and Gemini AI for intelligent difficulty calibration and personalized question recommendations.

---

## ✨ Features

- **Adaptive difficulty** — questions are chosen based on your current skill score (μ), which updates after every answer using an IRT-inspired model
- **Gemini AI calibration** — after enough player data accumulates, Gemini re-scores question difficulty based on real performance data and improves hints
- **Personalized recommendations** — Gemini analyzes your category weaknesses and biases question selection toward your blind spots
- **Auto question generation** — when you've seen 70%+ of the pool, Gemini silently generates 20 new questions in the background
- **Multiple game modes** — Free answer, Multiple Choice, Timed (15s countdown), Daily Challenge (same 10 questions for everyone each day), and 2-player local multiplayer
- **Leaderboard + profiles** — persistent scores, session history, accuracy by category, best streaks
- **Streak multipliers** — consecutive correct answers boost your skill gains

---

## 📂 Project Structure

```
nfl-trivia/
├── app.py                  # Flask app, all routes and game logic
├── wsgi.py                 # Gunicorn entry point
├── render.yaml             # Render deployment config
├── requirements.txt
├── .env                    # Local env vars (not committed)
├── data/
│   └── nfl_trivia.csv      # Question bank (id, question, answer, difficulty, category, hint)
├── src/
│   ├── trivia.py           # Skill model, question selection, answer checking
│   ├── gemini_ai.py        # Gemini-powered calibration, generation, personalization
│   └── models.py           # SQLAlchemy models (User, GameSession, QuestionRecord)
└── templates/
    ├── index.html          # Main game UI
    ├── auth.html           # Login / register
    ├── leaderboard.html    # Global leaderboard
    └── profile.html        # Per-user stats and session history
```

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
```
Get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com).

### 5. Run the app
```bash
python app.py
```
Visit `http://localhost:5000` and register an account to start playing.

---

## 🤖 How the AI Works

### Skill Model (IRT)
Each player has a skill score μ (starts at 0.0). After every answer:
- Correct answer on a hard question → big μ increase
- Wrong answer on an easy question → bigger μ decrease
- Response time also factors in — fast correct answers score better

Questions are selected from a band near your current μ, so difficulty always matches your level.

### Gemini Difficulty Calibration
After a game session ends, if enough player data has accumulated (30+ answers, 3+ per question), Gemini reviews each question's correct rate and the skill level of players who answered it. It suggests a revised difficulty score and an improved hint, which gets written back to `nfl_trivia.csv`.

### Personalized Recommendations
At the start of each game, your all-time category accuracy is sent to Gemini. It returns bias weights (e.g. "Draft: 2.5x, Super Bowl: 0.7x") that steer `choose_next()` toward categories where you need the most work.

### Auto Question Generation
Every time a question is loaded, the app checks if you've seen 70%+ of the question pool. If so, a background thread fires a Gemini API call to generate 20 new questions and append them to the CSV — completely silent, no delay to gameplay.

---

## 🌐 Deploying to Render

1. Push your code to GitHub
2. Create a new Web Service on [render.com](https://render.com) pointed at your repo
3. Render will use `render.yaml` to configure everything automatically
4. Add your `GEMINI_API_KEY` in the Render dashboard under **Environment**
5. `SECRET_KEY` and `DATABASE_URL` are handled automatically by `render.yaml`

---

## 🛠️ Tech Stack

- **Backend** — Python 3.11, Flask, Flask-Login, Flask-SQLAlchemy
- **Database** — PostgreSQL (Render) / SQLite (local dev)
- **AI** — Google Gemini 1.5 Pro
- **Frontend** — Vanilla JS, HTML/CSS (no framework)
- **Deployment** — Render (web service + managed Postgres)