import numpy as np
from config import DATA_DIR
import pandas as pd
import os
from data_loader import load_credentials, load_data, upload_data
from cosine_similarity import create_document_term_matrix, calculate_cosine_similarity
from integrate_lastfm import *

creds = load_credentials()

def save_similarity_matrix(matrix, feature_names, filepath):
    matrix = pd.DataFrame(matrix, index=feature_names, columns=feature_names)
    matrix.to_csv(filepath)
    return matrix

def generate_user_preferences(similarity_matrix, feature_names, num_users=10, songs_per_user=(50,100), top_similar=(5, 10)):
    """ Generate synthetic user data based on artist similarity. """
    sim_df = save_similarity_matrix(similarity_matrix, feature_names, DATA_DIR + '/song_similarity_matrix.csv')
    song_index = similarity_matrix.shape[0]
    # dictionary assigning masks of indices-to-song in the similarity matrix to users
    # requires selecting the data from the similarity matrix using the masks for semantics
    user_data = {}
    top_similar_n = np.random.randint(*top_similar)
    for user_id in range(num_users):
        num_songs = np.random.randint(*songs_per_user)
        # samples 40-50 starter songs
        starter_songs = np.random.choice(song_index, size=num_songs, replace=False)
        
        # selects these indices from the pandas df
        user_sim_df = sim_df.iloc[starter_songs,:]
        def select_top_similar(row, n=top_similar_n):
            '''
            selects top n similar songs to the starter song and adds it to user preferences
            '''

            curr_song_index = row.name # include starter song
            row = row.to_numpy()
            # excluding songs with same lyrics as starter
            non_starters = row != 1.0
            row = row[non_starters]
            top_n_indices = np.array([])
            if len(row) >= n:
                # selecting top 3 similarities
                top_n_indices_mask = np.argpartition(row, -n)[-n-1:-1]
                # sorting the top 3 similarities
                top_n_indices_mask = top_n_indices_mask[np.argsort(row[top_n_indices_mask])[::-1]]
                # returning the indices 
                top_n_indices = np.where(non_starters)[0][top_n_indices_mask]
            # adding original starter song to preferences
            top_n_indices = np.insert(top_n_indices, 0, curr_song_index)
            if user_id not in user_data:
                user_data[user_id] = top_n_indices.tolist()
            else:
                user_data[user_id].extend(top_n_indices.tolist())
        user_sim_df.apply(lambda row: select_top_similar(row), axis=1)

    return user_data

def transform_mask_to_songs(row_index, array, data):
    curr_user_index = row_index
    row = np.array(array[1])
    user_songs = data.iloc[row, :].copy(deep=True)
    user_songs['userID'] = curr_user_index

    return user_songs

def save_preferences(user_preferences):
    '''
    saves data locally and then uploads to google drive
    '''
    filename = DATA_DIR + '/user_preferences.csv'
    user_preferences.reset_index(drop=False, inplace=True)
    user_preferences.rename(columns={'index':'songID'}, inplace=True)
    user_preferences.to_csv(filename, index=False)
    print('Synthetic User Data Saved to Local File: ' + DATA_DIR + '/user_preferences.csv')
    upload_data(filename, creds=creds)
    print('Synthetic User Data Saved to Remote Drive.')

def integrate_lastfm(user_preferences):
  ### Adding additional LastFM Data for inference with NN Model ###
    user_preferences['lastfm_data'] =  user_preferences.progress_apply(fetch_and_parse, axis=1)
    user_preferences = user_preferences[user_preferences['lastfm_data'].notna()]
    user_preferences.reset_index(drop=True, inplace=True)
    track_details_df = pd.json_normalize(user_preferences['lastfm_data'])
    mixed = pd.concat(
        [user_preferences.drop(columns=['lastfm_data']), track_details_df], axis=1)
    print_tracks_skipped()
    return mixed


if __name__ == "__main__":
    file_id = '1EL4vYhO4A0Cgm8akBgAfDrWOGvtF6Xvo'
    file_name = 'millionsong_dataset.zip'
    data = load_data(file_id=file_id, file_name=file_name)

    ### Creating the Synthetic user preferences
    dtm, lyric_term_features = create_document_term_matrix(data)
    song_artist_index = list(data.index)
    similarity_matrix = calculate_cosine_similarity(dtm)
    user_preference_masks = generate_user_preferences(similarity_matrix, feature_names=song_artist_index)
    user_preferences = pd.concat([transform_mask_to_songs(row, array, data) for row, array in enumerate(user_preference_masks.items())], axis = 0)

    ### Connecting user data to additional LastFM features for model inference
    user_preferences = integrate_lastfm(user_preferences)
    save_preferences(user_preferences=user_preferences)
    
    

