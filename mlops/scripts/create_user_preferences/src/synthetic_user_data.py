import numpy as np
from config import DATA_DIR
import pandas as pd
from data_loader import load_data
from cosine_similarity import create_document_term_matrix, calculate_cosine_similarity

def save_similarity_matrix(matrix, feature_names, filepath):
    matrix = pd.DataFrame(matrix, index=feature_names, columns=feature_names)
    matrix.to_csv(filepath)
    return matrix

def generate_user_preferences(similarity_matrix, feature_names, num_users=1000, songs_per_user=(40, 50)):
    """ Generate synthetic user data based on artist similarity. """
    sim_df = save_similarity_matrix(similarity_matrix, feature_names, DATA_DIR + '/song_similarity_matrix.csv')
    song_index = similarity_matrix.shape[0]
    # dictionary assigning masks of indices-to-song in the similarity matrix to users
    # requires selecting the data from the similarity matrix using the masks for semantics
    user_data = {}
    for user_id in range(num_users):
        num_songs = np.random.randint(*songs_per_user)
        # samples 40-50 starter songs
        starter_songs = np.random.choice(song_index, size=num_songs, replace=False)
        # selects these indices from the pandas df
        user_sim_df = sim_df.iloc[starter_songs,:]
        def select_top_similar(row, n=3):
            '''
            returns top 3 similar song indices to the starter song sampled
            for each user to generate synthetic user preferences.

            It includes self with similarity of 1 (starter song + n similar songs)
            ex. n = 3
            returns: 160-200 songs per user * 1000 users = 16000-20000 rows
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
            user_data[user_id] = top_n_indices
        user_sim_df.apply(lambda row: select_top_similar(row), axis=1)

    return user_data

def transform_mask_to_songs(row_index, array, data):
    curr_user_index = row_index
    row = array[1]
    user_songs = data.iloc[row, :].copy(deep=True)
    user_songs['userID'] = curr_user_index

    return user_songs


if __name__ == "__main__":
    file_id = '1EL4vYhO4A0Cgm8akBgAfDrWOGvtF6Xvo'
    file_name = 'millionsong_dataset.zip'
    data = load_data(file_id=file_id, file_name=file_name)
    dtm, lyric_term_features = create_document_term_matrix(data['text'])
    song_artist_index = list(data.index)
    similarity_matrix = calculate_cosine_similarity(dtm)
    user_preference_masks = generate_user_preferences(similarity_matrix, feature_names=song_artist_index)
    pd.DataFrame.from_dict(user_preference_masks, orient='index').to_csv(DATA_DIR + '/user_preference_masks.csv')
    user_preferences = pd.concat([transform_mask_to_songs(row, array, data) for row, array in enumerate(user_preference_masks.items())], axis = 0)
    user_preferences.reset_index(drop=False, inplace=True)
    user_preferences.rename(columns={'index':'songID'}, inplace=True)
    user_preferences.to_csv(DATA_DIR + '/user_preferences.csv')
    print('Synthetic User Data Saved to Local File: ' + DATA_DIR + '/user_preferences.csv')

