import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from document_term_matrix import create_document_term_matrix
from data_loader import load_data
from config import DATA_DIR
def calculate_cosine_similarity(dtm):
    """ Calculate the cosine similarity matrix from a document-term matrix. """
    return cosine_similarity(dtm)

if __name__ == "__main__":
    data = load_data('million_song_dataset.csv')
    dtm, features = create_document_term_matrix(data['text'])
    similarity_matrix = calculate_cosine_similarity(dtm)