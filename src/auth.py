import requests
import os
from bs4 import BeautifulSoup
from dotenv import load_dotenv

def create_session() -> requests.Session:
    """Create an authenticated Kicktipp session."""
    load_dotenv()
    
    password = os.getenv("KICKTIPP_PASSWORD")
    email = os.getenv("KICKTIPP_EMAIL")
    
    if not password or not email:
        raise RuntimeError("Set KICKTIPP_EMAIL and KICKTIPP_PASSWORD in .env")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Origin': 'https://www.kicktipp.de',
        'Referer': 'https://www.kicktipp.de/info/profil/login',
    }
    
    s = requests.Session()
    s.headers.update(headers)
    
    login_url = "https://www.kicktipp.de/info/profil/login"
    resp = s.get(login_url)
    soup = BeautifulSoup(resp.text, 'html.parser')
    
    form = soup.find('form')
    if not form:
        raise RuntimeError("Login form not found")
    
    post_url = "https://www.kicktipp.de" + form.get("action")
    resp = s.post(post_url, data={
        "kennung": email,
        "passwort": password,
        "submitbutton": "Anmelden"
    })
    
    if "login" in resp.url:
        raise RuntimeError("Login failed")
    
    print("✓ Logged in to Kicktipp")
    return s