!pip install python-dotenv
# imports
import pandas as pd
import h5py
import os
from sqlalchemy import create_engine
import requests
import time
from dotenv import load_dotenv
import pandas as pd

df = pd.read_csv('/content/drive/MyDrive/CMPE-258: Team Neurobytes/Neurobytes/db/data/music_data.csv')
df.dropna(inplace=True)

import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch.optim as optim

# Encode categorical data
label_encoders = {}
unknown_label = 'unknown'  # Define an unknown label

for column in ['artist_name', 'tags', 'title']:
    le = LabelEncoder()

    # Get unique categories plus an 'unknown' category
    unique_categories = df[column].unique().tolist()
    # Add 'unknown' to the list of categories
    unique_categories.append(unknown_label)

    # Fit the LabelEncoder to these categories
    le.fit(unique_categories)
    df[column] = le.transform(df[column].astype(str))

    # Store the encoder
    label_encoders[column] = le


# Normalize numerical features
scaler = MinMaxScaler()
df[['duration', 'listeners', 'playcount']] = scaler.fit_transform(
    df[['duration', 'listeners', 'playcount']])

# Split data into features and target
X = df[['artist_name', 'tags', 'duration', 'listeners', 'playcount']]
y = df['title']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

class SongRecommender(nn.Module):
  def __init__(self):
      super(SongRecommender, self).__init__()
      self.fc1 = nn.Linear(5, 128)  # Adjust input features if needed
      self.fc2 = nn.Linear(128, 256)
      self.fc3 = nn.Linear(256, 128)
      # Output size = number of unique titles including 'unknown'
      # Add 1 for the 'unknown' label
      self.output = nn.Linear(128, len(y.unique()) + 1)

  def forward(self, x):
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = F.relu(self.fc3(x))
      x = self.output(x)
      return x


model = SongRecommender()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def train_model(model, X_train, y_train, X_test, y_test):
    train_loader = DataLoader(
        list(zip(X_train.values.astype(float), y_train)), batch_size=50, shuffle=True)
    test_loader = DataLoader(
        list(zip(X_test.values.astype(float), y_test)), batch_size=50, shuffle=False)

    model.train()
    for epoch in range(50):  # Number of epochs
        train_loss = 0
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(torch.tensor(features).float())
            # Ensure labels are long type
            loss = criterion(outputs, torch.tensor(labels).long())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        validation_loss = 0
        for features, labels in test_loader:
            outputs = model(torch.tensor(features).float())
            loss = criterion(outputs, torch.tensor(labels).long())
            validation_loss += loss.item()

        print(f'Epoch {epoch+1}, Training Loss: {train_loss / len(train_loader)}, Validation Loss: {validation_loss / len(test_loader)}')
train_model(model, X_train, y_train, X_test, y_test)
# save the model
torch.save(model.state_dict(), 'model.pth')
# load the model
model = SongRecommender()
def recommend_songs(model, input_features):
    model.eval()
    print(input_features)
    with torch.no_grad():
        try:
            artist_index = label_encoders['artist_name'].transform(
                [input_features['artist_name']])
        except ValueError:
            artist_index = label_encoders['artist_name'].transform(['unknown'])

        try:
            tags_index = label_encoders['tags'].transform(
                [input_features['tags']])
        except ValueError:
            tags_index = label_encoders['tags'].transform(['unknown'])

        # Create a DataFrame with feature names
        scaled_features = pd.DataFrame(
            [[input_features['duration'], input_features['listeners'],
                input_features['playcount']]],
            columns=['duration', 'listeners', 'playcount']
        )
        scaled_features = scaler.transform(scaled_features)[0]

        features = torch.tensor(
            [artist_index[0], tags_index[0], *scaled_features]).float().unsqueeze(0)
        predictions = model(features)
        top_5_values, top_5_indices = predictions.topk(5)
        recommended_song_ids = top_5_indices.squeeze().tolist()
        
        return label_encoders['title'].inverse_transform(recommended_song_ids)

import requests

def fetch_song_data(api_key, artist_name, track_name):
    url = "http://ws.audioscrobbler.com/2.0/"
    params = {
        'method': 'track.getInfo',
        'api_key': api_key,
        'artist': artist_name,
        'track': track_name,
        'format': 'json'
    }
    response = requests.get(url, params=params)
    print(response.content)
    return response.json() if response.status_code == 200 else {}


def parse_song_data(song_data):
    if song_data and 'track' in song_data:
        track = song_data['track']
        return {
            'artist_name': track['artist']['name'],
            'tags': ', '.join([tag['name'] for tag in track.get('toptags', {}).get('tag', [])]),
            'duration': float(track.get('duration', 0)),
            'listeners': int(track.get('listeners', 0)),
            'playcount': int(track.get('playcount', 0)),
            'album': track.get('album', {}).get('title', 'Unknown')
        }
    return {}
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('LASTFM_API_KEY')

artist_name = 'Lagy Gaga'
track_name = 'Poker Face'

# Fetch and parse song data
song_data = fetch_song_data(api_key, artist_name, track_name)
parsed_data = parse_song_data(song_data)

print(song_data)
# if the song is not found, or the tags column is empty, print a message
if not parsed_data or not parsed_data['tags']:
    print("Song not found or tags not available.")

else:
    recommend_songs(model, parsed_data)