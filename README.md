# 🏈 NFL Trivia ML Project

A personal project to explore **Python, pandas, and simple ML/AI concepts** by building an adaptive NFL trivia game.  
The app will track user answers and adjust question difficulty based on correctness and response time.

---

## 📂 Project Structure
---

nfl-trivia-ml/
│── data/
│ └── nfl_trivia.csv # Sample trivia questions
│── src/
│ └── trivia.py # Core functions (skill update, question selection)
│── pandaswarmup.py # Warm-up script for loading/querying CSV
│── README.md # Project description

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/JTrust-Process/NFL-Trivia.git
cd NFL-Trivia

2. Install dependencies
pip install -r requirements.txt

3. Run warm-up script
python pandaswarmup.py

🎯 Goals

✅ Day 1: Create sample CSV + warm-up script with pandas.

🔄 Day 2: Add fake responses + skill tracking function.

🔮 Future: Plot skill progression, adaptive difficulty, and interactive trivia game loop.

🛠️ Tech Stack

Python 3

pandas / numpy

matplotlib

scikit-learn (planned for adaptive models)

VS Code for development

GitHub for version control

📌 Next Steps

Implement skill update logic (update_skill function).

Simulate trivia sessions and visualize skill progression.

Build interactive CLI trivia game.

Explore ML approaches for adaptive difficulty.
