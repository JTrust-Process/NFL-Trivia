import pandas as pd

# Load CSV into a DataFrame
questions = pd.read_csv("nfl_trivia.csv")

pd.set_option("display.max_columns", None)  # show all columns
pd.set_option("display.width", None)

# Look at the first few rows
print(questions.head())

print()
# Example: filter all "Super Bowl" category questions
super_bowl_qs = questions[questions["category"] == "Super Bowl"]
print(super_bowl_qs)
