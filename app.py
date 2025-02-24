from flask import Flask, render_template, request, jsonify
import os
import json
from nba_predictor import NBAPredictor

app = Flask(__name__)
predictor = NBAPredictor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_teams', methods=['GET'])
def get_teams():
    """Get list of all NBA teams"""
    teams = predictor.get_all_teams()
    return jsonify(teams)

@app.route('/get_players', methods=['GET'])
def get_players():
    """Get list of all NBA players"""
    players = predictor.get_all_players()
    return jsonify(players)

@app.route('/predict', methods=['POST'])
def predict():
    """Predict match outcome and player stats"""
    data = request.json
    
    home_team = data.get('home_team')
    away_team = data.get('away_team')
    home_lineup = data.get('home_lineup', [])
    away_lineup = data.get('away_lineup', [])
    
    if not home_team or not away_team:
        return jsonify({'error': 'Home team and away team are required'}), 400
    
    if len(home_lineup) != 5 or len(away_lineup) != 5:
        return jsonify({'error': 'Each team must have exactly 5 players in lineup'}), 400
    
    prediction = predictor.predict_match(home_team, away_team, home_lineup, away_lineup)
    
    return jsonify(prediction)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok'})

def initialize():
    """Initialize the predictor on startup"""
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Create models directory if it doesn't exist 
    if not os.path.exists('models'):
        os.makedirs('models')
        
    # Load data
    predictor.load_data()
    
    # Check if models exist, if not train them
    model_files = [f for f in os.listdir('models') if f.endswith('_model.pkl')]
    
    if len(model_files) < len(predictor.models):
        print("Training new models...")
        predictor.train_models()
        predictor.save_models()
    else:
        print("Loading existing models...")
        predictor.load_models()

if __name__ == '__main__':
    initialize()
    app.run(debug=True)