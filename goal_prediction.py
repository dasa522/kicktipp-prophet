import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

url = "https://www.football-data.co.uk/mmz4281/2526/D1.csv"
url2 = "https://www.football-data.co.uk/mmz4281/2425/D1.csv"

df = pd.read_csv(url)

# Ensure Date is datetime and drop rows with invalid dates
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['Date'])

# Aggregate goals and games
home = df.groupby('HomeTeam').agg(
    HGF=('FTHG','sum'), HGA=('FTAG','sum'), HG=('HomeTeam','size')
)
away = df.groupby('AwayTeam').agg(
    AGF=('FTAG','sum'), AGA=('FTHG','sum'), AG=('AwayTeam','size')
)

teams = home.join(away, how='outer').fillna(0)
teams['GF'] = teams['HGF'] + teams['AGF']
teams['GA'] = teams['HGA'] + teams['AGA']
teams['G']  = teams['HG'] + teams['AG']

# League baselines
avg_home = df['FTHG'].mean()
avg_away = df['FTAG'].mean()
avg_all  = (df['FTHG'].sum() + df['FTAG'].sum()) / len(df)

# Rates
teams['rate_overall_scored']   = teams['GF'] / teams['G']
teams['rate_overall_conceded'] = teams['GA'] / teams['G']
teams['rate_home_scored']      = teams['HGF'] / teams['HG'].replace(0,np.nan)
teams['rate_home_conceded']    = teams['HGA'] / teams['HG'].replace(0,np.nan)
teams['rate_away_scored']      = teams['AGF'] / teams['AG'].replace(0,np.nan)
teams['rate_away_conceded']    = teams['AGA'] / teams['AG'].replace(0,np.nan)

# Shrinkage (blend venue with overall). k controls strength of shrinkage.
k = 1.1
teams['attack_home'] = (
    (teams['rate_home_scored'] * teams['HG']) + (k * teams['rate_overall_scored'])
) / (teams['HG'] + k)
teams['attack_away'] = (
    (teams['rate_away_scored'] * teams['AG']) + (k * teams['rate_overall_scored'])
) / (teams['AG'] + k)

teams['defense_home'] = (
    (teams['rate_home_conceded'] * teams['HG']) + (k * teams['rate_overall_conceded'])
) / (teams['HG'] + k)
teams['defense_away'] = (
    (teams['rate_away_conceded'] * teams['AG']) + (k * teams['rate_overall_conceded'])
) / (teams['AG'] + k)

# Convert to strengths relative to league (higher attack/defense = better)
teams['AttackStrengthHome'] = teams['attack_home'] / avg_home
teams['AttackStrengthAway'] = teams['attack_away'] / avg_away
teams['DefenseStrengthHome'] = avg_away / teams['defense_home']   # fewer conceded -> higher
teams['DefenseStrengthAway'] = avg_home / teams['defense_away']

# Example: expected goals using venue-aware strengths

def expected_goals(home_team, away_team, tbl, avg_home, avg_away, max_goals = 12):
    ha = tbl.loc[home_team, 'AttackStrengthHome']
    hd = tbl.loc[home_team, 'DefenseStrengthHome']
    aa = tbl.loc[away_team, 'AttackStrengthAway']
    ad = tbl.loc[away_team, 'DefenseStrengthAway']
    lam_home = ha * (1 / ad) * avg_home
    lam_away = aa * (1 / hd) * avg_away
    best = (0,0); best_p = 0.0
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            p = poisson.pmf(i, lam_home) * poisson.pmf(j, lam_away)
            if p > best_p:
                best_p = p; best = (i,j)
    return best 

lam_h, lam_a = expected_goals('Werder Bremen', 'Stuttgart', teams, avg_home, avg_away)

NAME_MAP = {
    "FC Bayern München": "Bayern Munich",
    "FSV Mainz 05": "Mainz",
    "Bor. Mönchengladbach": "M'gladbach",
    "1. FC Köln": "FC Koln",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "VfB Stuttgart": "Stuttgart",
    "Werder Bremen": "Werder Bremen",
    "SC Freiburg": "Freiburg",
    "RB Leipzig": "RB Leipzig",
    "Hamburger SV": "Hamburg",
    "Bayer 04 Leverkusen": "Leverkusen",
    "1. FC Union Berlin": "Union Berlin",
    "VfL Wolfsburg": "Wolfsburg",
    "FC St. Pauli": "St Pauli",
    "1. FC Heidenheim 1846": "Heidenheim",
    "1899 Hoffenheim": "Hoffenheim",
    "Borussia Dortmund": "Dortmund",
    "FC Augsburg": "Augsburg",
}

def normalize_team(name, teams_index):
    n = NAME_MAP.get(name, name)
    if n in teams_index:
        return n
    # fallback: strip accents and dots
    import unicodedata, re
    s = unicodedata.normalize('NFKD', n).encode('ascii', 'ignore').decode('ascii')
    s = re.sub(r"[.']", "", s).strip()
    if s in teams_index:
        return s
    raise KeyError(f"Team not found after normalization: {name} -> {n}")

def predict_matches(upcoming, teams, avg_home, avg_away):
    preds = []
    for home_team, away_team in upcoming:
        try:
            h = normalize_team(home_team, teams.index)
            a = normalize_team(away_team, teams.index)
            (home_score,away_score) = expected_goals(h, a, teams, avg_home, avg_away)
            preds.append({"home_team": home_team, "away_team": away_team, "home_score": home_score, "away_score": away_score})
        except KeyError as e:
            preds.append({"home_team": home_team, "away_team": away_team, "error": str(e)})
    return preds

def get_prediction(upcoming):
    return predict_matches(upcoming,teams, avg_home, avg_away)