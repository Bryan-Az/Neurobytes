import pandas as pd
from config import DATA_DIR, PROJECT_DIR
import os
import io
import google
from google.cloud import storage
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.oauth2 import credentials
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import socket

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.connect_ex(('localhost', port)) == 0

def load_credentials():
    """Load or create new credentials."""
    SCOPES = ['https://www.googleapis.com/auth/drive']
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    try:
        creds, _ = google.auth.default()
    except google.auth.exceptions.DefaultCredentialsError:
        # Run the flow using the client secrets file
        if not is_port_in_use(8080):
            print('In if in load creds')
            path_to_json = PROJECT_DIR + "/client_secrets.json"  # Path relative to your main application file
            print(path_to_json)
            flow = InstalledAppFlow.from_client_secrets_file(path_to_json, SCOPES)
            creds = flow.run_local_server(port=8080, access_type='offline', prompt='consent')
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
            creds = credentials.Credentials.from_authorized_user_file('token.json', SCOPES)
            if isinstance(creds, Credentials):
                print("creds is a valid Credentials object.")
            else:
                print("creds is not a valid Credentials object.")
        else:
            print('In else in load creds')
            # Load the credentials from token.json if they exist and check for expiry
            try:
                with open('token.json', 'r') as token:
                    creds = credentials.Credentials.from_authorized_user_file('token.json', SCOPES)
                if isinstance(creds, Credentials):
                    print("creds is a valid Credentials object.")
                else:
                    print("creds is not a valid Credentials object.")
            except FileNotFoundError:
                print("Token file not found. Please re-run the authentication flow.")
                raise
    return creds

def search_files(service, file_name):
    try:
        creds = load_credentials()
        print('credentials loaded')
        # Link PyDrive to use the credentials
        service = build('drive', 'v3', credentials=creds)
        print('auth with gdrive')

        results = service.files().list(
            pageSize=10,
            fields="nextPageToken, files(id, name)",
            q=f"name='{file_name}' and trashed=false"
        ).execute()
        items = results.get('files', [])

        if not items:
            print('No files found.')
        else:
            print('Files:')
            for item in items:
                print(u'{0} ({1})'.format(item['name'], item['id']))

    except Exception as e:
        print('An error occured in auth with google')
        print(e)
    return items

def load_data(file_id, file_name, n=10000):
    """
    Load data from a specified file within the data directory.
    Only a sample of 2000 rows will be used to load data
    as the original dataset is very large.
    Args:
    file_name (str): The name of the file to load.
    
    Returns:
    DataFrame: A pandas DataFrame containing the loaded data.
    """
    if os.path.exists(DATA_DIR + '/' + file_name):
        file_path = os.path.join(DATA_DIR, file_name)
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        sample_data = data.sample(n, replace=False).reset_index(drop=False)
        return sample_data
    else:
        try:
            creds = load_credentials()
            print('credentials loaded')
            # Link PyDrive to use the credentials
            service = build('drive', 'v3', credentials=creds)
            print('auth with gdrive')
            request = service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request, chunksize=204800)  # Adjust chunk size as needed
            done = False
            try:
                while not done:
                    status, done = downloader.next_chunk()
                    print("Download progress: {0}".format(status.progress() * 100))
                fh.seek(0)
                with open(DATA_DIR + '/'+ file_name, 'wb') as f:
                    f.write(fh.read())
                print(f"Download of '{DATA_DIR + '/'+ file_name}' complete.")
            except Exception as e:
                print(f"An error occurred: {e}")

                print('Data accessed from glink')
            file_path = os.path.join(DATA_DIR, file_name)
            data = pd.read_csv(file_path)
            print(f"Data loaded successfully from {file_path}")
            sample_data = data.sample(n, replace=False).reset_index(drop=False)
            return sample_data

        except Exception as e:
            print('An error occured in auth with google')
            print(e)