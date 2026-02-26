"""
Generates trivia questions from real NFL data using nfl-data-py.
Falls back gracefully if the library or network is unavailable.
"""
import pandas as pd


def generate_questions_from_nfl_data(seasons: list = None) -> list:
    """
    Pull real NFL stats and turn them into trivia questions.
    Uses nfl-data-py which fetches from nflverse (GitHub-hosted, no scraping needed).

    Returns a list of dicts ready to append to nfl_trivia.csv.
    """
    if seasons is None:
        seasons = [2022, 2023]

    try:
        import nfl_data_py as nfl
    except ImportError:
        print("nfl-data-py not installed — skipping auto question generation")
        return []

    questions = []

    # ── Passing stats ─────────────────────────────────────────────────────────
    try:
        passing = nfl.import_seasonal_data(seasons, s_type="REG")
        passing = passing[passing["season_type"] == "REG"] if "season_type" in passing.columns else passing

        for _, row in passing.nlargest(20, "passing_tds").iterrows():
            name   = row.get("player_display_name") or row.get("player_name", "")
            year   = int(row.get("season", 0))
            tds    = int(row.get("passing_tds", 0))
            yards  = int(row.get("passing_yards", 0))
            team   = row.get("recent_team") or row.get("team", "")
            if not name or not year:
                continue

            questions.append({
                "question": f"How many touchdown passes did {name} throw in the {year} regular season?",
                "answer": str(tds),
                "difficulty": 4,
                "category": "Records",
                "hint": f"He played for {team} that season and threw for {yards} yards",
            })
            questions.append({
                "question": f"How many passing yards did {name} throw for in the {year} regular season?",
                "answer": str(yards),
                "difficulty": 4,
                "category": "Records",
                "hint": f"He threw {tds} touchdown passes that year for {team}",
            })
    except Exception as e:
        print(f"Passing stats skipped: {e}")

    # ── Rushing stats ─────────────────────────────────────────────────────────
    try:
        rushing = nfl.import_seasonal_data(seasons, s_type="REG")
        for _, row in rushing.nlargest(15, "rushing_yards").iterrows():
            name   = row.get("player_display_name") or row.get("player_name", "")
            year   = int(row.get("season", 0))
            yards  = int(row.get("rushing_yards", 0))
            tds    = int(row.get("rushing_tds", 0))
            team   = row.get("recent_team") or row.get("team", "")
            if not name or not year or yards < 100:
                continue

            questions.append({
                "question": f"How many rushing yards did {name} gain in the {year} regular season?",
                "answer": str(yards),
                "difficulty": 4,
                "category": "Records",
                "hint": f"He scored {tds} rushing TDs that year playing for {team}",
            })
    except Exception as e:
        print(f"Rushing stats skipped: {e}")

    # ── Receiving stats ───────────────────────────────────────────────────────
    try:
        receiving = nfl.import_seasonal_data(seasons, s_type="REG")
        for _, row in receiving.nlargest(15, "receiving_yards").iterrows():
            name   = row.get("player_display_name") or row.get("player_name", "")
            year   = int(row.get("season", 0))
            yards  = int(row.get("receiving_yards", 0))
            recs   = int(row.get("receptions", 0))
            team   = row.get("recent_team") or row.get("team", "")
            if not name or not year or yards < 100:
                continue

            questions.append({
                "question": f"How many receiving yards did {name} record in the {year} regular season?",
                "answer": str(yards),
                "difficulty": 4,
                "category": "Records",
                "hint": f"He made {recs} receptions playing for {team}",
            })
    except Exception as e:
        print(f"Receiving stats skipped: {e}")

    # ── Schedule-based: Super Bowl winners ───────────────────────────────────
    try:
        schedule = nfl.import_schedules(seasons)
        sb_games = schedule[schedule["game_type"] == "SB"]
        for _, row in sb_games.iterrows():
            year = int(row.get("season", 0))
            home = row.get("home_team", "")
            away = row.get("away_team", "")
            home_score = row.get("home_score", 0)
            away_score = row.get("away_score", 0)
            if not (home and away and home_score is not None):
                continue
            winner = home if home_score > away_score else away
            loser  = away if home_score > away_score else home
            questions.append({
                "question": f"Which team won Super Bowl after the {year} NFL season?",
                "answer": _team_abbr_to_name(winner),
                "difficulty": 3,
                "category": "Super Bowl",
                "hint": f"They beat the {_team_abbr_to_name(loser)} in the championship game",
            })
    except Exception as e:
        print(f"Schedule stats skipped: {e}")

    print(f"nfl-data-py generated {len(questions)} candidate questions")
    return questions


def _team_abbr_to_name(abbr: str) -> str:
    """Map common NFL team abbreviations to full names."""
    mapping = {
        "KC": "Kansas City Chiefs", "SF": "San Francisco 49ers",
        "PHI": "Philadelphia Eagles", "DAL": "Dallas Cowboys",
        "NE": "New England Patriots", "BUF": "Buffalo Bills",
        "MIA": "Miami Dolphins", "NYJ": "New York Jets",
        "BAL": "Baltimore Ravens", "PIT": "Pittsburgh Steelers",
        "CLE": "Cleveland Browns", "CIN": "Cincinnati Bengals",
        "HOU": "Houston Texans", "IND": "Indianapolis Colts",
        "TEN": "Tennessee Titans", "JAX": "Jacksonville Jaguars",
        "DEN": "Denver Broncos", "LV": "Las Vegas Raiders",
        "LAC": "Los Angeles Chargers", "SEA": "Seattle Seahawks",
        "LAR": "Los Angeles Rams", "ARI": "Arizona Cardinals",
        "ATL": "Atlanta Falcons", "CAR": "Carolina Panthers",
        "NO": "New Orleans Saints", "TB": "Tampa Bay Buccaneers",
        "GB": "Green Bay Packers", "CHI": "Chicago Bears",
        "DET": "Detroit Lions", "MIN": "Minnesota Vikings",
        "NYG": "New York Giants", "WAS": "Washington Commanders",
    }
    return mapping.get(abbr.upper(), abbr)