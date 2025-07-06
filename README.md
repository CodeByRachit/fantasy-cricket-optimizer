# 🏏 Fantasy Cricket Team Optimizer

This project provides a complete pipeline to scrape, process, predict, and optimize fantasy cricket teams for the Indian T20 League (IPL) using machine learning and linear programming.

It includes:

✅ Real-time player data scraping from ESPN Cricinfo  
✅ Merging with historical batting, bowling, and venue performance data  
✅ Training an XGBoost regression model to predict fantasy points  
✅ Optimizing team selection under realistic rules  
✅ Generating a recommended playing XI with captain and vice-captain  
✅ Docker support for easy deployment  

---

## 📂 Project Structure

```plaintext
fantasy-cricket-optimizer/
├── satte.py
├── requirements.txt
├── Dockerfile
├── .gitignore
├── .gitattributes
├── LICENSE
├── README.md
├── data/
│   ├── Final_batter_ipl.csv
│   ├── bowler_ipl.csv
│   ├── wicketkeeper_ipl.csv
│   ├── batter_vs_bowler_head_to_head_2024.csv
│   ├── batter_venue_stats_2023_2024.csv
│   ├── bowler_performance_by_venue_2023_2024.csv
│   ├── teamwise_home_and_away.csv
│   ├── player_credits.csv
│   ├── Match_day_stadium.csv
│   └── SquadPlayerNames_IndianT20League.xlsx
├── models/
│   └── xgb_fantasy_model.pkl (generated after training)
└── outputs/
    └── FantasyPredictors_output.csv (generated after team optimization)
