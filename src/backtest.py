import pandas as pd
import argparse
from tqdm import tqdm
from src.data import load_bundesliga_data
from src.kicktipp_scoring import get_kicktipp_points
from models.dixon_coles import DixonColes
from models.poisson import PoissonModel

# Register all models/strategies you want to test
STRATEGIES = {
    #"poisson": PoissonModel,
    "dixonColes": DixonColes
}

def run_backtest(strategy_name: str, df: pd.DataFrame, min_train_size: int = 45):
    """
    Runs a backtest for a given strategy on a historical dataframe.
    Uses an expanding window to train the model and predict chronologically.
    """
    results = []
    
    # Sort by date to ensure order
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Iterate through each match day, starting after the initial training period
    for i in tqdm(range(min_train_size, len(df)), desc=f"Backtesting {strategy_name}"):
        # Train on all data *before* the current game
        train_df = df.iloc[:i]
        
        # The game to predict
        test_game = df.iloc[i]
        
        # Initialize and fit the model on the training data
        model = STRATEGIES[strategy_name]()
        model.fit(train_df)
        
        # Predict the score
        home_team = test_game['HomeTeam']
        away_team = test_game['AwayTeam']
        
        try:
            pred_home, pred_away = model.predict(home_team, away_team)
            
            # Get actual score
            actual_home = int(test_game['FTHG'])
            actual_away = int(test_game['FTAG'])
            
            # Calculate points
            points = get_kicktipp_points(pred_home, pred_away, actual_home, actual_away)
            
            results.append({
                'date': test_game['Date'],
                'home_team': home_team,
                'away_team': away_team,
                'prediction': f"{pred_home}-{pred_away}",
                'actual': f"{actual_home}-{actual_away}",
                'points': points
            })
        except KeyError as e:
            # Handle cases where a team might not be in the training set yet
            pass
            
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Backtest Kicktipp prediction strategies.")
    parser.add_argument(
        "--season", 
        type=str, 
        default="2526", 
        help="The season to backtest on (e.g., '2425')."
    )
    args = parser.parse_args()

    print(f"Loading data for season {args.season}...")
    # We need at least two seasons: one to train on initially, one to test
    season_to_test = int(args.season)
    prev_season = str(season_to_test - 101) # '2526' -> '2425'
    #prev2_season = str(int(prev_season) - 101) # '2526' -> '2425'
    
    df = load_bundesliga_data(seasons=[prev_season, args.season])
    
    print("\n" + "="*50)
    print(f"BACKTEST RESULTS FOR SEASON {args.season}")
    print("="*50)
    
    all_results = {}
    for name in STRATEGIES.keys():
        result_df = run_backtest(name, df.copy())
        total_points = result_df['points'].sum()
        avg_points = result_df['points'].mean()
        all_results[name] = total_points
        
        print(f"\nStrategy: {name}")
        print(f"  - Total Points: {total_points}")
        print(f"  - Avg Points/Game: {avg_points:.2f}")
        print("  - Points Distribution:")
        print(result_df['points'].value_counts(normalize=True).sort_index().to_string())

    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    sorted_results = sorted(all_results.items(), key=lambda item: item[1], reverse=True)
    for name, points in sorted_results:
        print(f"  - {name}: {points} points")


if __name__ == "__main__":
    main()