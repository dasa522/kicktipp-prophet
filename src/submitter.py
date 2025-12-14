from typing import List, Dict
from bs4 import BeautifulSoup
import requests


def submit_tips(
    session: requests.Session,
    community: str,
    matches: List[Dict],
    predictions: List[Dict]
) -> bool:
    """
    Submit predictions to Kicktipp.
    
    Args:
        session: Authenticated requests session
        community: Kicktipp community name
        matches: List of match dicts with home_team, away_team, home_field, away_field
        predictions: List of prediction dicts with home_team, away_team, home_score, away_score
    
    Returns:
        True if submission was successful
    """
    # Build lookup: (home_team, away_team) -> (home_score, away_score)
    pred_lookup = {}
    for p in predictions:
        if "error" not in p:
            pred_lookup[(p['home_team'], p['away_team'])] = (p['home_score'], p['away_score'])
    
    # Fetch the form page to get hidden fields
    tip_url = f"https://www.kicktipp.de/{community}/tippabgabe"
    resp = session.get(tip_url)
    soup = BeautifulSoup(resp.text, 'html.parser')
    
    form = soup.find('form')
    if not form:
        print("❌ Tip form not found on tippabgabe page")
        return False
    
    post_url = "https://www.kicktipp.de" + form.get("action")
    
    # Collect hidden fields
    payload = {}
    for inp in form.find_all('input', {'type': 'hidden'}):
        name = inp.get('name')
        value = inp.get('value', '')
        if name:
            payload[name] = value
    
    # Fill tips for each match
    filled = 0
    for m in matches:
        key = (m['home_team'], m['away_team'])
        if key in pred_lookup:
            h, a = pred_lookup[key]
            payload[m['home_field']] = str(h)
            payload[m['away_field']] = str(a)
            filled += 1
            print(f"   ✓ {m['home_team']} {h} - {a} {m['away_team']}")
        else:
            print(f"   ⚠️ No prediction for {m['home_team']} vs {m['away_team']}, skipping.")
    
    if filled == 0:
        print("❌ No tips to submit")
        return False
    
    print(f"\nSubmitting {filled} tips...")
    submit_resp = session.post(post_url, data=payload)
    
    if submit_resp.status_code == 200:
        print("✅ Tipps gespeichert successfully!")
        return True
    else:
        print(f"❌ Submit failed (status: {submit_resp.status_code})")
        sub_soup = BeautifulSoup(submit_resp.text, 'html.parser')
        errors = sub_soup.find_all(class_=['messages', 'res_error', 'ct_error'])
        for e in errors:
            print(f"   Error: {e.get_text(strip=True)}")
        return False