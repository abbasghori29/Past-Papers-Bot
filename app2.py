from flask import Flask, jsonify,redirect,url_for
from flask_session import Session
from msal import ConfidentialClientApplication
from dotenv import load_dotenv
import os
import requests

app = Flask(__name__)

TENANT_ID = "178774fc-26a7-49ed-977e-0feb6a7d866b"
CLIENT_ID = "be9fab77-1cd5-45c1-91f3-81371dfae656"
CLIENT_SECRET = "DGU8Q~Bc7-_lQc3k~_KQsTIpfsCPxl2ybsybMaFx"
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
REDIRECT_URI = "http://localhost:5000/getAToken"
SCOPE = ["https://graph.microsoft.com/.default"]
SESSION_TYPE = "filesystem"
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SESSION_TYPE'] = SESSION_TYPE
Session(app)


def get_access_token():
    app = ConfidentialClientApplication(
        client_id=CLIENT_ID,
        client_credential=CLIENT_SECRET,
        authority=AUTHORITY
    )
    
    result = app.acquire_token_silent(SCOPE, account=None)
    if not result:
        result = app.acquire_token_for_client(scopes=SCOPE)
    
    return result.get('access_token')

@app.route('/onedrive')
def list_onedrive_contents():
    access_token = get_access_token()
    if not access_token:
        return "Failed to get access token", 400

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json"
    }
    url = "https://graph.microsoft.com/v1.0/users/talhahashmi940@gmail.com/drive"
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        return f"Error: {response.status_code}, {response.text}"
    return jsonify(response.json())


if __name__ == '__main__':
    app.run(debug=True) 
