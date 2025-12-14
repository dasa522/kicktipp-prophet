from abc import ABC, abstractmethod
from typing import List, Tuple, Dict

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