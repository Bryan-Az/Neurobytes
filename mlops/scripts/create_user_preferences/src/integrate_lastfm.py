from dotenv import load_dotenv
# load LASTFM_API_KEY from .env
import requests
import os

def fetch_data(api_key, method, params):
    base_url = "http://ws.audioscrobbler.com/2.0/"
    params['api_key'] = api_key
    params['method'] = method
    params['format'] = 'json'
    response = requests.get(base_url, params=params)
    return response.json()


def get_artist_info(api_key, artist_name):
    params = {'artist': artist_name}
    return fetch_data(api_key, 'artist.getInfo', params)


def get_track_info(api_key, artist_name, track_name):
    params = {'artist': artist_name, 'track': track_name}
    return fetch_data(api_key, 'track.getInfo', params)


def batch_fetch_data(api_key, items, fetch_function, sleep_time=1):
    results = []
    for item in items:
        result = fetch_function(api_key, *item)
        results.append(result)
        # time.sleep(sleep_time)
    return results

api_key = os.getenv('LASTFM_API_KEY')


def fetch_lastfm_data(api_key, artist_name, track_name):
    base_url = "http://ws.audioscrobbler.com/2.0/"
    params = {
        'method': 'track.getInfo',
        'api_key': api_key,
        'artist': artist_name,
        'track': track_name,
        'format': 'json'
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200 and response.text.strip():
        return response.json()
    else:
        return None


def parse_lastfm_data(data):
    if data and 'track' in data:
        track = data['track']
        return {
            'listeners': track.get('listeners', '0'),
            'playcount': track.get('playcount', '0'),
            'tags': ', '.join(tag['name'] for tag in track.get('toptags', {}).get('tag', [])),
        }
    return None

from tqdm import tqdm
tqdm.pandas()

load_dotenv()
api_key = os.getenv('LASTFM_API_KEY')
tracks_skipped = 0

def print_tracks_skipped():
    print(f"Tracks skipped: {tracks_skipped}")


def fetch_and_parse(row):
    global tracks_skipped
    data = fetch_lastfm_data(api_key, row['artist'], row['song'])
    if data is None:
        tracks_skipped += 1
        return None
    parsed_data = parse_lastfm_data(data)
    if parsed_data is None:
        tracks_skipped += 1
    return parsed_data