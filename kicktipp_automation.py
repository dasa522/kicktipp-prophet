import requests as re
import os 
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from goal_prediction import get_prediction 

load_dotenv()

# 1. Verify Password is loaded
password = os.getenv("KICKTIPP_PASSWORD")
if not password:
    print("Error: KICKTIPP_PASSWORD is not set in your .env file")
    exit()

login_url = "https://www.kicktipp.de/info/profil/login"

# 2. Add standard browser headers (Origin and Referer are crucial)
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Origin': 'https://www.kicktipp.de',
    'Referer': 'https://www.kicktipp.de/info/profil/login',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
    'Accept-Language': 'de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7'
}

s = re.Session()
s.headers.update(headers)

# 3. Get the login page to set cookies
print("Fetching login page...")
initial_response = s.get(login_url)
soup = BeautifulSoup(initial_response.text, 'html.parser')

form = soup.find('form')
if form:
    post_url = "https://www.kicktipp.de" + form.get("action")
    print(f"Posting to: {post_url}")
    
    form_data = {
        "kennung": "davidschneider2305@gmail.com",
        "passwort": password,
        "submitbutton": "Anmelden"
    }

    # 4. Perform Login
    response = s.post(post_url, data=form_data)
    
    # 5. Check results
    if response.url != login_url and "login" not in response.url:
        print("✓ Login successful!")
        print(f"Redirected to: {response.url}")
    else:
        print("✗ Login failed.")
        print(f"Current URL: {response.url}")
        
        # 6. Extract error message from page
        error_soup = BeautifulSoup(response.text, 'html.parser')
        # Kicktipp usually puts errors in a div with class 'messages' or 'ct_error'
        errors = error_soup.find_all(class_=['messages', 'res_error', 'ct_error'])
        if errors:
            print("\nError messages found on page:")
            for error in errors:
                print(f"- {error.get_text(strip=True)}")
        else:
            print("No specific error message found, but login failed.")
tip_url = "https://www.kicktipp.de/lovers/tippabgabe"
response = s.get(tip_url)

soup = BeautifulSoup(response.text, 'html.parser')
matches = []
upcoming = []

# Find all rows in the table
rows = soup.find_all('tr')

print(f"\nScanning {len(rows)} rows for matches...")

for row in rows:
    # Find the input fields for home and away goals
    # Updated to match your debug output: "heimTipp" and "gastTipp"
    home_input = row.find('input', {'name': lambda x: x and 'heimTipp' in x})
    away_input = row.find('input', {'name': lambda x: x and 'gastTipp' in x})

    if home_input and away_input:
        # We found a row with a match!
        
        # 1. Extract the form field names (Crucial for submission later)
        home_field_name = home_input['name']
        away_field_name = away_input['name']
        
        # 2. Extract Team Names
        # Kicktipp structure is usually: Date | HomeTeam | AwayTeam | Inputs
        cols = row.find_all('td')
        
        # Try to find specific classes first (standard Kicktipp themes)
        home_cell = row.find('td', class_='heim')
        away_cell = row.find('td', class_='gast')
        
        # Fallback to column position if classes aren't found
        if not home_cell and len(cols) > 2:
            home_cell = cols[1] 
        if not away_cell and len(cols) > 2:
            away_cell = cols[2]

        if home_cell and away_cell:
            home_team = home_cell.get_text(strip=True)
            away_team = away_cell.get_text(strip=True)
            
            match_data = {
                'home_team': home_team,
                'away_team': away_team,
                'home_field': home_field_name,
                'away_field': away_field_name
            }
            matches.append(match_data)
            upcoming.append((home_team,away_team))

print(f"\nSuccessfully extracted {len(matches)} matches.")
preds = get_prediction(upcoming)  # list of dicts with score etc., using name normalization inside
# Create a quick lookup: (home, away) -> (h, a)
pred_lookup = {}
for p in preds:
    if "error" in p: 
        print(f"Prediction error for {p['home_team']} vs {p['away_team']}: {p['error']}")
        continue
    h = p["home_score"]
    a = p["away_score"]
    pred_lookup[(p['home_team'], p['away_team'])] = (h, a)

# Find the form and collect hidden fields
form = soup.find('form')
if not form:
    raise RuntimeError("Tip form not found on tippabgabe page")
post_url = "https://www.kicktipp.de" + form.get("action")

payload = {}
# Include all hidden inputs (tipperId, spieltagIndex, bonus, tippsaisonId, etc.)
for inp in form.find_all('input', {'type': 'hidden'}):
    name = inp.get('name'); value = inp.get('value', '')
    if name: payload[name] = value

# Fill tips for each match from predictions
filled = 0
for m in matches:
    key = (m['home_team'], m['away_team'])
    if key in pred_lookup:
        h, a = pred_lookup[key]
        payload[m['home_field']] = str(h)
        payload[m['away_field']] = str(a)
        filled += 1
    else:
        print(f"No prediction for {key}, skipping.")

print(f"Prepared payload with {filled} tips. Submitting...")
submit_resp = s.post(post_url, data=payload)
if submit_resp.status_code == 200 and "Tipps gespeichert" in submit_resp.text:
    print("✓ Tipps gespeichert successfully.")
else:
    print(f"Submit status: {submit_resp.status_code}")
    # Try to extract any error messages
    sub_soup = BeautifulSoup(submit_resp.text, 'html.parser')
    errors = sub_soup.find_all(class_=['messages', 'res_error', 'ct_error'])
    for e in errors:
        print(f"Error: {e.get_text(strip=True)}")
