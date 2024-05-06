import numpy as np
from config import DATA_DIR
import pandas as pd
from data_loader import load_data
from cosine_similarity import create_document_term_matrix, calculate_cosine_similarity

def generate_user_preferences(similarity_matrix, num_users=1000, songs_per_user=(100, 200)):
    """ Generate synthetic user data based on artist similarity. """
    num_artists = similarity_matrix.shape[0]
    user_data = {}
    for user_id in range(num_users):
        num_songs = np.random.randint(*songs_per_user)
        sampled_artists = np.random.choice(num_artists, size=np.random.randint(10, 21), replace=False)
        user_data[user_id] = np.random.choice(sampled_artists, size=num_songs, replace=True).tolist()
    return user_data

if __name__ == "__main__":
    data = load_data('million_song_dataset.csv')
    dtm, features = create_document_term_matrix(data['text'])
    similarity_matrix = calculate_cosine_similarity(dtm)
    user_preferences = generate_user_preferences(similarity_matrix)
    pd.DataFrame.from_dict(user_preferences, orient='index').to_csv(DATA_DIR + '/user_preferences.csv')