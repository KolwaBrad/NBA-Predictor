import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class NBAPredictor:
    def __init__(self):
        self.data_dir = 'data'
        self.models_dir = 'models'
        self.team_abbrev = None
        self.player_data = {}
        self.team_data = {}
        self.lineup_data = {}
        self.team_name_map = {}
        self.models = {
            'team_score': None,
            'player_points': None,
            'player_rebounds': None,
            'player_assists': None,
            'player_steals': None,
            'player_blocks': None,
            'player_turnovers': None
        }
        
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
            
        self._load_team_abbreviations()
        
    def _load_team_abbreviations(self):
        """Load team name to abbreviation mapping"""
        df = pd.read_csv(os.path.join(self.data_dir, 'team_name_abbreviation.csv'))
        self.team_abbrev = df
        self.team_name_map = dict(zip(df['Team'], df['TEAM_ABBREVIATION']))
        self.team_abbr_to_name = dict(zip(df['TEAM_ABBREVIATION'], df['Team']))
        
    def load_data(self):
        """Load all necessary data from CSV files"""
        seasons = ['2019_20', '2020_21', '2021_22', '2022_23', '2023_24', '2024_25']
        
        for season in seasons:
            file_path = os.path.join(self.data_dir, f'nba_player_stats_{season}.csv')
            if os.path.exists(file_path):
                self.player_data[season] = pd.read_csv(file_path)
                
        for season in seasons:
            file_path = os.path.join(self.data_dir, f'nba_team_stats_{season}.csv')
            if os.path.exists(file_path):
                self.team_data[season] = pd.read_csv(file_path)
                
        for season in seasons:
            file_path = os.path.join(self.data_dir, f'nba_lineups_{season}_per_game.csv')
            if os.path.exists(file_path):
                self.lineup_data[season] = pd.read_csv(file_path)
                
        print(f"Loaded data for {len(self.player_data)} seasons")
    
    def prepare_team_features(self):
        """Prepare features for team score prediction"""
        team_features = []
        team_scores = []
        
        for season, data in self.team_data.items():
            if data is not None and not data.empty:
                for _, row in data.iterrows():
                    team_name = row['Team']
                    team_abbr = self.team_name_map.get(team_name)
                    
                    if team_abbr:
                        features = {
                            'WIN_PCT': row['WIN%'] if 'WIN%' in row else 0.5,
                            'PTS_PER_GAME': row['PTS'],
                            'FG_PCT': row['FG%'],
                            'FG3_PCT': row['3P%'],
                            'FT_PCT': row['FT%'],
                            'REB': row['REB'],
                            'AST': row['AST'],
                            'TOV': row['TOV'],
                            'STL': row['STL'],
                            'BLK': row['BLK']
                        }
                        team_features.append(features)
                        team_scores.append(row['PTS'])
        
        return pd.DataFrame(team_features), team_scores
    
    def prepare_player_features(self):
        """Prepare features for player stat predictions"""
        player_features = []
        player_points = []
        player_rebounds = []
        player_assists = []
        player_steals = []
        player_blocks = []
        player_turnovers = []
        
        for season, data in self.player_data.items():
            if data is not None and not data.empty:
                for _, row in data.iterrows():
                    if row['MIN'] > 0:
                        features = {
                            'AGE': row['AGE'],
                            'MIN': row['MIN'] / row['GP'],
                            'FG_PCT': row['FG_PCT'],
                            'FG3_PCT': row['FG3_PCT'],
                            'FT_PCT': row['FT_PCT'],
                            'W_PCT': row['W_PCT']
                        }
                        player_features.append(features)
                        player_points.append(row['PTS'] / row['GP'])
                        player_rebounds.append(row['REB'] / row['GP'])
                        player_assists.append(row['AST'] / row['GP'])
                        player_steals.append(row['STL'] / row['GP'])
                        player_blocks.append(row['BLK'] / row['GP'])
                        player_turnovers.append(row['TOV'] / row['GP'])
        
        return (pd.DataFrame(player_features), 
                player_points, player_rebounds, player_assists, 
                player_steals, player_blocks, player_turnovers)
    
    def train_models(self):
        """Train all prediction models"""
        team_X, team_y = self.prepare_team_features()
        team_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        team_pipeline.fit(team_X, team_y)
        self.models['team_score'] = team_pipeline
        
        player_X, player_points, player_reb, player_ast, player_stl, player_blk, player_tov = self.prepare_player_features()
        
        points_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        points_pipeline.fit(player_X, player_points)
        self.models['player_points'] = points_pipeline
        
        reb_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        reb_pipeline.fit(player_X, player_reb)
        self.models['player_rebounds'] = reb_pipeline
        
        ast_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        ast_pipeline.fit(player_X, player_ast)
        self.models['player_assists'] = ast_pipeline
        
        stl_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        stl_pipeline.fit(player_X, player_stl)
        self.models['player_steals'] = stl_pipeline
        
        blk_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        blk_pipeline.fit(player_X, player_blk)
        self.models['player_blocks'] = blk_pipeline
        
        tov_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        tov_pipeline.fit(player_X, player_tov)
        self.models['player_turnovers'] = tov_pipeline
        
        print("All models trained successfully")
    
    def save_models(self):
        """Save all trained models to disk"""
        for model_name, model in self.models.items():
            if model is not None:
                model_path = os.path.join(self.models_dir, f'{model_name}_model.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
        print("All models saved to disk")
    
    def load_models(self):
        """Load all saved models from disk"""
        for model_name in self.models.keys():
            model_path = os.path.join(self.models_dir, f'{model_name}_model.pkl')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
        print("All available models loaded from disk")
    
    def get_player_last_season_stats(self, player_name):
        """Get player stats from the most recent season available"""
        latest_seasons = ['2024_25', '2023_24', '2022_23', '2021_22', '2020_21', '2019_20']
        
        for season in latest_seasons:
            if season in self.player_data and self.player_data[season] is not None:
                player_df = self.player_data[season]
                if 'PLAYER_NAME' in player_df.columns:
                    player_info = player_df[player_df['PLAYER_NAME'] == player_name]
                    if not player_info.empty:
                        return player_info.iloc[0].to_dict()
        
        return {
            'AGE': 25,
            'MIN': 20,
            'FG_PCT': 0.45,
            'FG3_PCT': 0.35,
            'FT_PCT': 0.75,
            'W_PCT': 0.5
        }
    
    def get_team_last_season_stats(self, team_name):
        """Get team stats from the most recent season available"""
        if team_name in self.team_abbr_to_name:
            team_name = self.team_abbr_to_name[team_name]
        
        latest_seasons = ['2024_25', '2023_24', '2022_23', '2021_22', '2020_21', '2019_20']
        
        for season in latest_seasons:
            if season in self.team_data and self.team_data[season] is not None:
                team_df = self.team_data[season]
                if 'Team' in team_df.columns:
                    team_info = team_df[team_df['Team'] == team_name]
                    if not team_info.empty:
                        return team_info.iloc[0].to_dict()
        
        return {
            'WIN%': 0.5,
            'PTS': 110,
            'FG%': 0.45,
            '3P%': 0.35,
            'FT%': 0.75,
            'REB': 45,
            'AST': 25,
            'TOV': 15,
            'STL': 8,
            'BLK': 5
        }
    
    def predict_match(self, home_team, away_team, home_lineup, away_lineup, home_bench, away_bench):
        """Predict match outcome and stats using both starting and bench players"""
        home_team_stats = self.get_team_last_season_stats(home_team)
        away_team_stats = self.get_team_last_season_stats(away_team)
        
        home_score_features = {
            'WIN_PCT': home_team_stats.get('WIN%', 0.5),
            'PTS_PER_GAME': home_team_stats.get('PTS', 110),
            'FG_PCT': home_team_stats.get('FG%', 0.45),
            'FG3_PCT': home_team_stats.get('3P%', 0.35),
            'FT_PCT': home_team_stats.get('FT%', 0.75),
            'REB': home_team_stats.get('REB', 45),
            'AST': home_team_stats.get('AST', 25),
            'TOV': home_team_stats.get('TOV', 15),
            'STL': home_team_stats.get('STL', 8),
            'BLK': home_team_stats.get('BLK', 5)
        }
        
        away_score_features = {
            'WIN_PCT': away_team_stats.get('WIN%', 0.5),
            'PTS_PER_GAME': away_team_stats.get('PTS', 110),
            'FG_PCT': away_team_stats.get('FG%', 0.45),
            'FG3_PCT': away_team_stats.get('3P%', 0.35),
            'FT_PCT': away_team_stats.get('FT%', 0.75),
            'REB': away_team_stats.get('REB', 45),
            'AST': away_team_stats.get('AST', 25),
            'TOV': away_team_stats.get('TOV', 15),
            'STL': away_team_stats.get('STL', 8),
            'BLK': away_team_stats.get('BLK', 5)
        }
        
        # Predict team scores (adding home court advantage)
        predicted_home_score = self.models['team_score'].predict(pd.DataFrame([home_score_features]))[0] + 3.5
        predicted_away_score = self.models['team_score'].predict(pd.DataFrame([away_score_features]))[0]
        
        predicted_home_score = max(80, min(130, predicted_home_score))
        predicted_away_score = max(80, min(130, predicted_away_score))
        
        # Bench multiplier to reflect lower playing time for substitutes
        bench_factor = 0.7
        
        def process_player(player_name, predicted_score, is_bench=False):
            player_last_stats = self.get_player_last_season_stats(player_name)
            player_features = {
                'AGE': player_last_stats.get('AGE', 25),
            'MIN': player_last_stats.get('MIN', 25) / player_last_stats.get('GP', 70),
            'FG_PCT': player_last_stats.get('FG_PCT', 0.45),
            'FG3_PCT': player_last_stats.get('FG3_PCT', 0.35),
            'FT_PCT': player_last_stats.get('FT_PCT', 0.75),
            'W_PCT': player_last_stats.get('W_PCT', 0.5)
                            }
            df_features = pd.DataFrame([player_features])
    
            points = self.models['player_points'].predict(df_features)[0]
            rebounds = self.models['player_rebounds'].predict(df_features)[0]
            assists = self.models['player_assists'].predict(df_features)[0]
            steals = self.models['player_steals'].predict(df_features)[0]
            blocks = self.models['player_blocks'].predict(df_features)[0]
            turnovers = self.models['player_turnovers'].predict(df_features)[0]
    
            pace_factor = predicted_score / 110
            points *= pace_factor
            rebounds *= pace_factor
            assists *= pace_factor
    
            if is_bench:
                bench_factor = 0.7
                points *= bench_factor
                rebounds *= bench_factor
                assists *= bench_factor
    
            points = max(0, min(55, points))
            rebounds = max(0, min(20, rebounds))
            assists = max(0, min(15, assists))
            steals = max(0, min(5, steals))
            blocks = max(0, min(5, blocks))
            turnovers = max(0, min(8, turnovers))
    
            return {
                'name': player_name,
                'points': round(points, 1),
                'rebounds': round(rebounds, 1),
                'assists': round(assists, 1),
                'steals': round(steals, 1),
                'blocks': round(blocks, 1),
                'turnovers': round(turnovers, 1),
                'role': 'bench' if is_bench else 'starter'
                    }

        
        # Process starters and bench for home team
        home_players = [process_player(player, predicted_home_score, False) for player in home_lineup]
        home_bench_players = [process_player(player, predicted_home_score, True) for player in home_bench]
        home_all = home_players + home_bench_players
        
        # Process starters and bench for away team
        away_players = [process_player(player, predicted_away_score, False) for player in away_lineup]
        away_bench_players = [process_player(player, predicted_away_score, True) for player in away_bench]
        away_all = away_players + away_bench_players
        
        # Normalize points so that the total matches the predicted team score
        def normalize_player_points(player_list, team_score):
            total_points = sum(p['points'] for p in player_list)
            if total_points > 0:
                ratio = team_score / total_points
                for p in player_list:
                    p['points'] = round(p['points'] * ratio, 1)
            return player_list
        
        home_all = normalize_player_points(home_all, predicted_home_score)
        away_all = normalize_player_points(away_all, predicted_away_score)
        
        # Split back into starters and bench
        home_starters = [p for p in home_all if p['role'] == 'starter']
        home_bench = [p for p in home_all if p['role'] == 'bench']
        away_starters = [p for p in away_all if p['role'] == 'starter']
        away_bench = [p for p in away_all if p['role'] == 'bench']
        
        # Compute team totals (summing over all players)
        def team_totals(player_list):
            return {
                'points': round(sum(p['points'] for p in player_list), 1),
                'rebounds': round(sum(p['rebounds'] for p in player_list), 1),
                'assists': round(sum(p['assists'] for p in player_list), 1),
                'steals': round(sum(p['steals'] for p in player_list), 1),
                'blocks': round(sum(p['blocks'] for p in player_list), 1),
                'turnovers': round(sum(p['turnovers'] for p in player_list), 1)
            }
        
        home_totals = team_totals(home_all)
        away_totals = team_totals(away_all)
        
        winner = home_team if predicted_home_score > predicted_away_score else away_team
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'predicted_home_score': round(predicted_home_score, 1),
            'predicted_away_score': round(predicted_away_score, 1),
            'winner': winner,
            'home_player_stats': {
                'starters': home_starters,
                'bench': home_bench
            },
            'away_player_stats': {
                'starters': away_starters,
                'bench': away_bench
            },
            'home_team_stats': home_totals,
            'away_team_stats': away_totals
        }
    
    def get_all_players(self):
        """Get a list of all players from the most recent season"""
        players = set()
        latest_seasons = ['2024_25', '2023_24', '2022_23', '2021_22', '2020_21', '2019_20']
        
        for season in latest_seasons:
            if season in self.player_data and self.player_data[season] is not None:
                player_df = self.player_data[season]
                if 'PLAYER_NAME' in player_df.columns:
                    season_players = player_df['PLAYER_NAME'].tolist()
                    players.update(season_players)
        
        return sorted(list(players))
    
    def get_all_teams(self):
        """Get a list of all team names"""
        return list(self.team_name_map.keys())

if __name__ == "__main__":
    predictor = NBAPredictor()
    predictor.load_data()
    predictor.train_models()
    predictor.save_models()
    print("NBA prediction models trained and saved successfully!")
