import pandas as pd
import numpy as np
from scipy.stats import poisson
from models.base import PredictionModel

# Kicktipp -> football-data.co.uk name mapping
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

class PoissonModel(PredictionModel):
    """Poisson-based prediction using attack/defense strengths."""
    
    name = "poisson"
    
    def __init__(self, shrinkage_k: float = 1.5, max_goals: int = 12):
        self.k = shrinkage_k
        self.max_goals = max_goals
        self.teams = None
        self.avg_home = None
        self.avg_away = None
    
    def fit(self, df: pd.DataFrame) -> None:
        """Compute team strengths from historical match data."""
        # Aggregate goals and games
        home = df.groupby('HomeTeam').agg(
            HGF=('FTHG', 'sum'), HGA=('FTAG', 'sum'), HG=('HomeTeam', 'size')
        )
        away = df.groupby('AwayTeam').agg(
            AGF=('FTAG', 'sum'), AGA=('FTHG', 'sum'), AG=('AwayTeam', 'size')
        )
        
        teams = home.join(away, how='outer').fillna(0)
        teams['GF'] = teams['HGF'] + teams['AGF']
        teams['GA'] = teams['HGA'] + teams['AGA']
        teams['G'] = teams['HG'] + teams['AG']
        
        # League baselines
        self.avg_home = df['FTHG'].mean()
        self.avg_away = df['FTAG'].mean()
        
        # Rates
        teams['rate_overall_scored'] = teams['GF'] / teams['G']
        teams['rate_overall_conceded'] = teams['GA'] / teams['G']
        teams['rate_home_scored'] = teams['HGF'] / teams['HG'].replace(0, np.nan)
        teams['rate_home_conceded'] = teams['HGA'] / teams['HG'].replace(0, np.nan)
        teams['rate_away_scored'] = teams['AGF'] / teams['AG'].replace(0, np.nan)
        teams['rate_away_conceded'] = teams['AGA'] / teams['AG'].replace(0, np.nan)
        
        # Shrinkage
        k = self.k
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
        
        # Strengths relative to league
        teams['AttackStrengthHome'] = teams['attack_home'] / self.avg_home
        teams['AttackStrengthAway'] = teams['attack_away'] / self.avg_away
        teams['DefenseStrengthHome'] = self.avg_away / teams['defense_home']
        teams['DefenseStrengthAway'] = self.avg_home / teams['defense_away']
        
        self.teams = teams
    
    def _normalize_team(self, name: str) -> str:
        """Map Kicktipp team names to football-data names."""
        n = NAME_MAP.get(name, name)
        if n in self.teams.index:
            return n
        import unicodedata, re
        s = unicodedata.normalize('NFKD', n).encode('ascii', 'ignore').decode('ascii')
        s = re.sub(r"[.']", "", s).strip()
        if s in self.teams.index:
            return s
        raise KeyError(f"Team not found: {name} -> {n}")
    
    def predict(self, home_team: str, away_team: str) -> tuple[int, int]:
        """Predict most likely score for a match."""
        h = self._normalize_team(home_team)
        a = self._normalize_team(away_team)
        
        ha = self.teams.loc[h, 'AttackStrengthHome']
        hd = self.teams.loc[h, 'DefenseStrengthHome']
        aa = self.teams.loc[a, 'AttackStrengthAway']
        ad = self.teams.loc[a, 'DefenseStrengthAway']
        
        lam_home = ha * (1 / ad) * self.avg_home
        lam_away = aa * (1 / hd) * self.avg_away
        
        best = (0, 0)
        best_p = 0.0
        for i in range(self.max_goals + 1):
            for j in range(self.max_goals + 1):
                p = poisson.pmf(i, lam_home) * poisson.pmf(j, lam_away)
                if p > best_p:
                    best_p = p
                    best = (i, j)
        return best