from abc import ABC, abstractmethod
from typing import List, Tuple, Dict

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

class PredictionModel(ABC):
    """Base class for all prediction models."""
    
    name: str = "base"
    
    @abstractmethod
    def fit(self, df) -> None:
        """Train/fit the model on historical data."""
        pass
    
    @abstractmethod
    def predict(self, home_team: str, away_team: str) -> Tuple[int, int]:
        """Predict the score for a single match."""
        pass
    def predict_proba(self, home_team: str, away_team: str) -> Dict[str, float]:
        """
        Predict probabilities for common betting markets.
        Default implementation raises an error, must be overridden.
        """
        raise NotImplementedError(f"The model '{self.name}' does not support probability predictions.")
    
    def predict_matches(self, matches: List[Tuple[str, str]]) -> List[Dict]:
        """Predict scores for multiple matches."""
        results = []
        for home, away in matches:
            try:
                h, a = self.predict(home, away)
                results.append({
                    "home_team": home,
                    "away_team": away,
                    "home_score": h,
                    "away_score": a
                })
            except KeyError as e:
                results.append({
                    "home_team": home,
                    "away_team": away,
                    "error": str(e)
                })
        return results

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