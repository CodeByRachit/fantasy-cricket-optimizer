import time
import requests
import csv
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from bs4 import BeautifulSoup
import tempfile
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os
import joblib
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpBinary
import re
from selenium.webdriver.chrome.service import Service

chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in headless mode (no GUI)
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems
chrome_options.add_argument(
   'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
)
chrome_options.add_argument("Accept-Language: en-US,en;q=0.9")
chrome_options.add_argument("Accept-Encoding: gzip, deflate, br")
chrome_options.add_argument("Connection: keep-alive")
userdata_dir = tempfile.mkdtemp()
chrome_options.add_argument(f'--user-data-dir={userdata_dir}')
url = "https://www.espncricinfo.com/series/ipl-2025-1449924/most-valuable-players"
driver=webdriver.Chrome(options=chrome_options, service=Service('/usr/local/bin/chromedriver'))
driver.get(url)
time.sleep(5)


html=driver.page_source

soup=BeautifulSoup(html , "html.parser")

#table=soup.find("table" , class_="ds-w-full ds-table ds-table-md ds-table-auto")
table=soup.find("table")
file1=open("data/impact_points.csv" , "w" , encoding ="utf-8")

writer=csv.writer(file1)
writer.writerow(["Name","Team","Total points","average points","match played","total runs ","total wickets"])
rows=table.find_all("tr")
for row in rows:
    cells= row.find_all("td")
    data = [cell.text.strip() for cell in cells]
    
    writer.writerow(data)

file1.close()
driver.quit()

batting = pd.read_csv(r"data/Final_batter_ipl.csv")
bowling = pd.read_csv(r"data/bowler_ipl.csv")
wk = pd.read_csv(r"data/wicketkeeper_ipl.csv")
h2h = pd.read_csv(r"data/batter_vs_bowler_head_to_head_2024.csv")
batting_venue = pd.read_csv(r"data/batter_venue_stats_2023_2024.csv")
bowling_venue = pd.read_csv(r"data/bowler_performance_by_venue_2023_2024.csv")
home_away = pd.read_csv(r"data/teamwise_home_and_away.csv")
roles = pd.read_csv(r"data/player_credits.csv")
impact = pd.read_csv(r"data/impact_points.csv")
impact['Name']= impact['Name'].str.replace(r'^\d+', '', regex=True).str.strip()

batting.rename(columns={"Player": "player"}, inplace=True)
bowling.rename(columns={"Player": "player"}, inplace=True)
wk.rename(columns={"Player": "player"}, inplace=True)
roles.rename(columns={"Player Name": "player"}, inplace=True)
impact.rename(columns={"Name": "player"}, inplace=True)

# Step 2: Combine Batting, Bowling, WK stats
combined = pd.merge(batting, bowling, on="player", how="outer")
combined = pd.merge(combined, wk, on="player", how="outer")

# Step 3: Add role information and impact points
combined = pd.merge(combined, roles, on="player", how="left")
combined = pd.merge(combined, impact, on="player", how="left")

# Step 4: Fill missing values
combined.fillna(0, inplace=True)

# Step 5: Feature Engineering
combined['total_dismissals'] = combined['Catches'] + combined['Stumping']

# Step 6: Save the training data
combined.to_csv("data/combined_training_data.csv", index=False)
print("✅ Combined training data saved to data/combined_training_data.csv")

df = pd.read_csv(r"data/combined_training_data.csv")

# Define features and target (using Average points instead of Total points)
features = [
    'Runs', 'Avg_x', 'S/R', 'Wkts', 'Avg_y', 'E/R', 'total_dismissals', 'average points', 'Credits'
]
target = 'average points'

# Drop rows with missing target
df = df.dropna(subset=[target])

X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.1, max_depth=5)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)

print(f"✅ Model Trained! MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# Create the 'models' directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save the model
joblib.dump(model, "models/xgb_fantasy_model.pkl")
print("💾 Model saved to models/xgb_fantasy_model.pkl")

squad_player = "/app/data/SquadPlayerNames_IndianT20League.xlsx"
xls = pd.ExcelFile(squad_player)
sheet_names = xls.sheet_names

latest_lineup = pd.read_excel(xls, sheet_name=sheet_names[-1])
CURRENT_MATCH_NO = int(sheet_names[-1][6::])

latest_lineup.to_csv("data/latest_lineup.csv", index=False)
latest_lineup_1 = pd.read_csv(r"data/latest_lineup.csv")

model = joblib.load("models/xgb_fantasy_model.pkl")
today_df = pd.read_csv(r"data/latest_lineup.csv")
today_df.rename(columns={"Player Name": "player", "Credits": "credits", "Player Type": "role", "Team": "team"}, inplace=True)

# Filter only playing players and X-Factor substitutes
today_df = today_df[today_df['IsPlaying'].isin(["PLAYING", "X_FACTOR_SUBSTITUTE"])]

# Load the training data
training_data = pd.read_csv(r"data/combined_training_data.csv")

# Load venue info and extract stadium for current match
venue_info = pd.read_csv(r"data/Match_day_stadium.csv")
match_venue = venue_info[venue_info['Match No'] == CURRENT_MATCH_NO].values[0][1]


# Load player venue stats
batter_venue = pd.read_csv(r"data/batter_venue_stats_2023_2024.csv")
bowler_venue = pd.read_csv(r"data/bowler_performance_by_venue_2023_2024.csv")

batter_venue['bat_venue_score'] = batter_venue['strike_rate'] / (batter_venue['dismissals'] + 1)
bowler_venue['bowl_venue_score'] = bowler_venue['wickets'] / (bowler_venue['economy'] + 1)

# Filter for current venue
batter_venue_filtered = batter_venue[batter_venue['venue'] == match_venue][['batter', 'bat_venue_score']]
bowler_venue_filtered = bowler_venue[bowler_venue['venue'] == match_venue][['bowler', 'bowl_venue_score']]

# Merge today's players with training stats
merged = pd.merge(today_df, training_data, on="player", how="left")
merged = merged.drop_duplicates(subset="player")

# Merge venue scores
merged = pd.merge(merged, batter_venue_filtered, how='left', left_on='player', right_on='batter')
merged = pd.merge(merged, bowler_venue_filtered, how='left', left_on='player', right_on='bowler')
merged.fillna(0, inplace=True)

# Calculate total venue-based boost
merged['venue_boost'] = merged['bat_venue_score'] + merged['bowl_venue_score']

# Predict fantasy points
features = ['Runs', 'Avg_x', 'S/R', 'Wkts', 'Avg_y', 'E/R', 'total_dismissals', 'average points', 'Credits']
merged['predicted_points'] = model.predict(merged[features])

# Apply base boost
merged['adjusted_predicted_points'] = merged['predicted_points'] * (1 + 0.15 * merged['venue_boost'])



# Optimization
prob = LpProblem("Fantasy_Team_Selection", LpMaximize)
player_vars = {i: LpVariable(f'player_{i}', cat=LpBinary) for i in merged.index}

prob += lpSum([player_vars[i] * merged.loc[i, 'adjusted_predicted_points'] for i in merged.index])

prob += lpSum([player_vars[i] * merged.loc[i, 'credits'] for i in merged.index]) <= 100

# Minimum players from each role
role_requirements = {'WK': 1, 'BAT': 3, 'AR': 2, 'BOWL': 3}
for role, minimum in role_requirements.items():
    prob += lpSum([player_vars[i] for i in merged.index if merged.loc[i, 'role'] == role]) >= minimum

# At least 1 player from each team
teams = merged['team'].unique()
for team in teams:
    prob += lpSum([player_vars[i] for i in merged.index if merged.loc[i, 'team'] == team]) >= 1

# Enforce team balance: 5:6 or 6:5
team_1, team_2 = teams[0], teams[1]
team_1_count = lpSum([player_vars[i] for i in merged.index if merged.loc[i, 'team'] == team_1])
team_2_count = lpSum([player_vars[i] for i in merged.index if merged.loc[i, 'team'] == team_2])
prob += team_1_count - team_2_count <= 1
prob += team_2_count - team_1_count <= 1


prob += lpSum([player_vars[i] for i in merged.index]) == 12

# Solve
prob.solve()

# Output
os.makedirs("outputs", exist_ok=True)
selected_indices = [i for i in merged.index if player_vars[i].varValue is not None and round(player_vars[i].varValue) == 1]
selected = merged.loc[selected_indices]
selected_sorted = selected.sort_values(by='adjusted_predicted_points', ascending=False).reset_index(drop=True)

# Assign C with preference to WK or BAT, VC as before
wk_bat_players = selected_sorted[selected_sorted['role'].isin(['WK', 'BAT','ALL'])].sort_values(by='adjusted_predicted_points', ascending=False)
if len(wk_bat_players) >= 1:  # Check for at least 1 WK/BAT
    cap = wk_bat_players.iloc[0]['player']
else:
    cap = selected_sorted.iloc[0]['player']  # Fallback to highest predicted

vc = selected_sorted[selected_sorted['player'] != cap].iloc[0]['player'] # VC is 2nd highest, excluding captain
# Mark C/VC or leave blank
selected_sorted['C/VC'] = selected_sorted['player'].apply(
    lambda x: 'C' if x == cap else ('VC' if x == vc else '')
)

# Replace empty strings with None (so it shows as null in CSV)
selected_sorted['C/VC'] = selected_sorted['C/VC'].replace('', None)


# Save output
final_output = selected_sorted[['player', 'team', 'C/VC']]
final_output.to_csv("outputs/FantasyPredictors_output.csv", index=False)


print("\n🏏 Selected Playing XI (with Captain and Vice-Captain):")
print(selected_sorted[['player', 'team', 'C/VC']])