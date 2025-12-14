import pandas as pd
from typing import List

def load_bundesliga_data(seasons: List[str] = None) -> pd.DataFrame:
    """
    Load Bundesliga match data from football-data.co.uk.
    
    Args:
        seasons: List of seasons like ['2425', '2526']. Defaults to current season.
    
    Returns:
        DataFrame with match results.
    """
    if seasons is None:
        seasons = ['2526']
    
    base_url = "https://www.football-data.co.uk/mmz4281/{}/D1.csv"
    
    dfs = []
    for season in seasons:
        url = base_url.format(season)
        try:
            df = pd.read_csv(url)
            df['Season'] = season
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not load season {season}: {e}")
    
    if not dfs:
        raise RuntimeError("No data loaded")
    
    combined = pd.concat(dfs, ignore_index=True)
    combined['Date'] = pd.to_datetime(combined['Date'], dayfirst=True, errors='coerce')
    combined = combined.dropna(subset=['Date', 'FTHG', 'FTAG'])
    
    return combined