import gradio as gr
import torch
import torch.nn as nn
from joblib import load

# Define the same neural network model
class ImprovedSongRecommender(nn.Module):
    def __init__(self, input_size, num_titles):
        super(ImprovedSongRecommender, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.output = nn.Linear(128, num_titles)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.output(x)
        return x

# Load the trained model
model_path = "models/improved_model.pth"
num_unique_titles = 4855  

model = ImprovedSongRecommender(input_size=2, num_titles=num_unique_titles)  
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Load the label encoders and scaler
label_encoders_path = "data/new_label_encoders.joblib"
scaler_path = "data/new_scaler.joblib"

label_encoders = load(label_encoders_path)
scaler = load(scaler_path)

# Create a mapping from encoded indices to actual song titles
index_to_song_title = {index: title for index, title in enumerate(label_encoders['title'].classes_)}

def encode_input(tags, artist_name):
    tags = tags.strip().replace('\n', '')
    artist_name = artist_name.strip().replace('\n', '')

    try:
        encoded_tags = label_encoders['tags'].transform([tags])[0]
    except ValueError:
        encoded_tags = label_encoders['tags'].transform(['unknown'])[0]

    if artist_name:
        try:
            encoded_artist = label_encoders['artist_name'].transform([artist_name])[0]
        except ValueError:
            encoded_artist = label_encoders['artist_name'].transform(['unknown'])[0]
    else:
        encoded_artist = label_encoders['artist_name'].transform(['unknown'])[0]

    return [encoded_tags, encoded_artist]

def recommend_songs(tags, artist_name):
    encoded_input = encode_input(tags, artist_name)
    input_tensor = torch.tensor([encoded_input]).float()
    
    with torch.no_grad():
        output = model(input_tensor)
    
    recommendations_indices = torch.topk(output, 5).indices.squeeze().tolist()
    recommendations = [index_to_song_title.get(idx, "Unknown song") for idx in recommendations_indices]
    
    formatted_output = [f"Recommendation {i+1}: {rec}" for i, rec in enumerate(recommendations)]
    return formatted_output

# Set up the Gradio interface
interface = gr.Interface(
    fn=recommend_songs,
    inputs=[gr.Textbox(lines=1, placeholder="Enter Tags (e.g., rock)"), gr.Textbox(lines=1, placeholder="Enter Artist Name (optional)")],
    outputs=gr.Textbox(label="Recommendations"),
    title="Music Recommendation System",
    description="Enter tags and (optionally) artist name to get music recommendations."
)

interface.launch()
