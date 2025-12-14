from typing import List, Tuple, Dict
from bs4 import BeautifulSoup
import requests

def get_upcoming_matches(session: requests.Session, community: str = "lovers") -> List[Dict]:
    """
    Scrape upcoming matches from Kicktipp tippabgabe page.
    
    Returns:
        List of dicts with home_team, away_team, home_field, away_field.
    """
    url = f"https://www.kicktipp.de/{community}/tippabgabe"
    resp = session.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')
    
    matches = []
    for row in soup.find_all('tr'):
        home_input = row.find('input', {'name': lambda x: x and 'heimTipp' in x})
        away_input = row.find('input', {'name': lambda x: x and 'gastTipp' in x})
        
        if home_input and away_input:
            cols = row.find_all('td')
            home_cell = row.find('td', class_='heim') or (cols[1] if len(cols) > 2 else None)
            away_cell = row.find('td', class_='gast') or (cols[2] if len(cols) > 2 else None)
            
            if home_cell and away_cell:
                matches.append({
                    'home_team': home_cell.get_text(strip=True),
                    'away_team': away_cell.get_text(strip=True),
                    'home_field': home_input['name'],
                    'away_field': away_input['name'],
                })
    
    return matches