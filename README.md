# Kicktipp Predictor ⚽🔮

Automatically predict Bundesliga match scores and submit tips to [Kicktipp](https://www.kicktipp.de). Beat your friends with data-driven predictions!

## Features

- 🤖 **Automated predictions** using statistical models
- 📊 **Poisson regression model** with home/away strength factors
- 🔄 **Automatic tip submission** to Kicktipp
- 🧩 **Extensible architecture** - easily add new prediction models
- ⚙️ **Configurable** via YAML - no code changes needed

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/kicktipp-predictor.git
cd kicktipp-predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure credentials

```bash
cp .env.example .env
```

Edit `.env` with your Kicktipp login:

```
KICKTIPP_EMAIL="your-email@example.com"
KICKTIPP_PASSWORD="your-password"
```

### 4. Configure your community

Edit `config.yaml`:

```yaml
# Your Kicktipp community name (from URL: kicktipp.de/<community>/tippabgabe)
community: "your-community-name"

# Which prediction model to use
model: "poisson"
```

### 5. Run predictions

```bash
# Show predictions only
python main.py

# Show predictions AND submit to Kicktipp
python main.py --submit
```

## Example Output

```
Loading historical data...
Loaded 117 matches
Fitting poisson model...
Logging in to Kicktipp...
✓ Logged in to Kicktipp
Found 9 upcoming matches

==================================================
PREDICTIONS
==================================================
⚽ FC Bayern München 3 - 1 Borussia Dortmund
⚽ RB Leipzig 2 - 0 VfB Stuttgart
⚽ Bayer 04 Leverkusen 2 - 1 Eintracht Frankfurt
...

💡 Run with --submit to submit these predictions to Kicktipp
```

## Project Structure

```
kicktipp-predictor/
├── main.py              # Entry point
├── config.yaml          # User configuration
├── requirements.txt     # Python dependencies
├── .env.example         # Template for credentials
│
├── src/                 # Core functionality
│   ├── auth.py          # Kicktipp login
│   ├── scraper.py       # Scrape upcoming matches
│   ├── submitter.py     # Submit tips
│   └── data.py          # Load historical match data
│
├── models/              # Prediction models
│   ├── base.py          # Abstract base class
│   └── poisson.py       # Poisson regression model
│
└── tests/               # Unit tests
```

## Available Models

| Model         | Description                                               | Config Key    |
|---------------|----------------------------------------------------------|--------------|
| **Poisson**   | Poisson regression with venue-aware attack/defense strengths | `poisson`    |
| **Dixon-Coles** | Time-decayed Poisson with correlation for low scores (draws, 0-0, 1-0, 0-1) | `dixonColes` |

### Poisson Model

The default model uses [Poisson regression](https://en.wikipedia.org/wiki/Poisson_regression) to predict match scores:

1. **Calculates team strengths** from historical data:
   - Attack strength (home/away)
   - Defense strength (home/away)
2. **Applies shrinkage** to handle teams with few games
3. **Predicts expected goals** using:
   ```
   λ_home = AttackStrengthHome × (1 / DefenseStrengthAway) × LeagueAvgHomeGoals
   λ_away = AttackStrengthAway × (1 / DefenseStrengthHome) × LeagueAvgAwayGoals
   ```
4. **Finds most probable score** by maximizing $P(\text{home}=i) \times P(\text{away}=j)$

#### Configuration

```yaml
poisson:
  shrinkage_k: 1.5    # Higher = more regression to mean (good for early season)
  max_goals: 12       # Maximum goals to consider per team
```

### Dixon-Coles Model

The [Dixon-Coles model](https://www.sportingintelligence.com/wp-content/uploads/2010/08/Dixon-Coles-1997.pdf) is an extension of Poisson regression that:

- **Adds a correlation parameter ($\rho$)** to better model low-scoring draws and upsets
- **Uses time decay** so recent matches have more influence
- **Regularizes parameters** to avoid overfitting

#### Configuration

```yaml
dixonColes:
  time_decay_alpha: 0.001      # How quickly older matches lose influence
  regularization_lambda: 0.01  # L2 penalty strength for parameter shrinkage
  max_goals: 12                # Maximum goals to consider per team
```

#### How it works

- Optimizes attack/defense strengths, home advantage, and $\rho$ using maximum likelihood
- Predicts the most probable score using the adjusted joint probability

#### Usage

To use Dixon-Coles, set in your `config.yaml`:

```yaml
model: "dixonColes"

dixonColes:
  time_decay_alpha: 0.001
  regularization_lambda: 0.01
  max_goals: 12
```

---

## Adding Your Own Model

1. Create a new file in `models/`:

```python
# models/my_model.py
from models.base import PredictionModel

class MyModel(PredictionModel):
    name = "my_model"
    
    def __init__(self, my_param: float = 1.0):
        self.my_param = my_param
    
    def fit(self, df):
        # Train on historical data (DataFrame with FTHG, FTAG, HomeTeam, AwayTeam)
        pass
    
    def predict(self, home_team: str, away_team: str) -> tuple[int, int]:
        # Return predicted (home_goals, away_goals)
        return (1, 1)
```

2. Register it in `main.py`:

```python
from models.my_model import MyModel

MODELS = {
    "poisson": PoissonModel,
    "my_model": MyModel,  # Add this line
}
```

3. Use it in `config.yaml`:

```yaml
model: "my_model"

my_model:
  my_param: 2.0
```

## Data Source

Historical match data is fetched from [football-data.co.uk](https://www.football-data.co.uk/), which provides:

- Match results (home/away goals)
- Betting odds from multiple bookmakers
- Match statistics

Configure which seasons to use:

```yaml
data:
  seasons:
    - "2425"  # 2024/25 season
    - "2526"  # 2025/26 season
```

## Requirements

- Python 3.10+
- Dependencies: `pandas`, `numpy`, `scipy`, `requests`, `beautifulsoup4`, `python-dotenv`, `pyyaml`

## Troubleshooting

### "Team not found" errors

Team names differ between Kicktipp and football-data.co.uk. The model includes a mapping for Bundesliga teams, but you may need to extend it in `models/poisson.py`:

```python
NAME_MAP = {
    "Kicktipp Name": "football-data Name",
    # Add missing teams here
}
```

### Login fails

- Check your credentials in `.env`
- Ensure there are no extra quotes or spaces
- Kicktipp may temporarily block automated logins - try again later

### No matches found

- Verify your community name in `config.yaml`
- Ensure there are upcoming matches on the Tippabgabe page

## Contributing

Contributions are welcome! Ideas for new models:

- **Elo rating system**
- **Machine learning** (XGBoost, Random Forest)
- **Ensemble methods** (combine multiple models)
- **Betting odds integration** (use market odds as features)


## Disclaimer

This tool is for educational purposes. Use responsibly and in accordance with Kicktipp's terms of service. Automated betting predictions do not guarantee success.

---

Made with ❤️ for Kicktipp enthusiasts who want to beat their friends with data science.