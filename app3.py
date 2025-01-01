import os
from O365 import Account
from langchain_community.document_loaders.onedrive import OneDriveLoader

# Set environment variables for authentication
os.environ['O365_CLIENT_ID'] = 'be9fab77-1cd5-45c1-91f3-81371dfae656'
os.environ['O365_CLIENT_SECRET'] = '~jw8Q~BhZBf~jGVA1rVXfFeBRrijZRneNWcbwdoA'

# Define the required scopes
scopes = [
    'offline_access',
    'Files.Read',
    'Files.Read.All',
    'Sites.Read.All'
]

# Token file path
token_path = "C:/Users/Abbas/Desktop/pp project test/credentials/o365_token.txt"

# Check if the token file exists
if not os.path.exists(token_path):
    raise FileNotFoundError(f"Token file not found: {token_path}. Please ensure it exists.")

# Initialize the account with scopes and token path
credentials = (os.environ['O365_CLIENT_ID'], os.environ['O365_CLIENT_SECRET'])
account = Account(credentials, scopes=scopes, token_path=token_path)

# Authenticate if not already authenticated
if not account.is_authenticated:
    account.authenticate()

# Initialize the OneDriveLoader
loader = OneDriveLoader(
    drive_id='2c96b4e8cbcb67b5',
    folder_path='AmirEjaz',
    auth_with_token=True
)

# Load documents
documents = loader.load()
print(documents)
