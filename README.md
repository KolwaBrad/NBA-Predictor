# NBA Predictor

A machine learning-powered system that predicst NBA matchups outcomes as well as team and player stats for each matchup.

## Features

- 7 models to predict each avenue -6 for player and 1 for team stats 
- Interactive web interface built with Flask
- Data-driven insights from 6 last NBA seasons including 2024-25 season

## Tech Stack

- Python 3.8+
- Flask web framework
- Scikit-learn for machine learning
- Bootstrap for frontend styling

## Installation

1. Clone the repository:
```bash
git clone https://github.com/KolwaBrad/NBA-Predictor
cd nba-predictor
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # For Unix/macOS
venv\Scripts\activate     # For Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```



## Project Structure

```
nba-predictor/
├── app.py
├── nba_predictor.py
├── data/
│   ├── nba_player_stats_2019_20.csv
│   ├── nba_player_stats_2020_21.csv
│   ├── nba_player_stats_2021_22.csv
│   ├── nba_player_stats_2022_23.csv
│   ├── nba_player_stats_2023_24.csv
│   ├── nba_player_stats_2024_25.csv
│   ├── nba_lineups_2019_20_per_game.csv
│   ├── nba_lineups_2020_21_per_game.csv
│   ├── nba_lineups_2021_22_per_game.csv
│   ├── nba_lineups_2022_23_per_game.csv
│   ├── nba_lineups_2023_24_per_game.csv
│   ├── nba_lineups_2024_25_per_game.csv
│   ├── nba_team_stats_2019_20.csv
│   ├── nba_team_stats_2020_21.csv
│   ├── nba_team_stats_2021_22.csv
│   ├── nba_team_stats_2022_23.csv
│   ├── nba_team_stats_2023_24.csv
│   ├── nba_team_stats_2024_25.csv
│   └── team_name_abbreviation.csv
├── models/
└── templates/
    └── index.html

```

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to `http://127.0.0.1:5000`

3. Enter home and away teams and starting players for each



## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Website - Soon

Project Link: [https://github.com/KolwaBrad/NBA-Predictor](https://github.com/KolwaBrad/NBA-Predictor)