import yaml
import argparse
from src.auth import create_session
from src.scraper import get_upcoming_matches
from src.submitter import submit_tips
from src.data import load_bundesliga_data
from models.poisson import PoissonModel

# Add more models here as you create them
MODELS = {
    "poisson": PoissonModel,
}


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Kicktipp Predictor")
    parser.add_argument("--submit", action="store_true", help="Submit predictions to Kicktipp")
    args = parser.parse_args()
    
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Load data
    print("Loading historical data...")
    df = load_bundesliga_data(config['data']['seasons'])
    print(f"Loaded {len(df)} matches")
    
    # Initialize model
    model_name = config['model']
    model_config = config.get(model_name, {})
    model = MODELS[model_name](**model_config)
    
    print(f"Fitting {model_name} model...")
    model.fit(df)
    
    # Get upcoming matches
    print("Logging in to Kicktipp...")
    session = create_session()
    matches = get_upcoming_matches(session, config['community'])
    print(f"Found {len(matches)} upcoming matches")
    
    # Predict
    upcoming = [(m['home_team'], m['away_team']) for m in matches]
    predictions = model.predict_matches(upcoming)
    
    # Display predictions
    print("\n" + "=" * 50)
    print("PREDICTIONS")
    print("=" * 50)
    for p in predictions:
        if "error" in p:
            print(f"❌ {p['home_team']} vs {p['away_team']}: {p['error']}")
        else:
            print(f"⚽ {p['home_team']} {p['home_score']} - {p['away_score']} {p['away_team']}")
    
    # Submit if flag is set
    if args.submit:
        print("\n" + "=" * 50)
        print("SUBMITTING TIPS")
        print("=" * 50)
        submit_tips(session, config['community'], matches, predictions)
    else:
        print("\n💡 Run with --submit to submit these predictions to Kicktipp")


if __name__ == "__main__":
    main()