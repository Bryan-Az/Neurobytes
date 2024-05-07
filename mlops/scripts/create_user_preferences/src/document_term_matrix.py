import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from config import DATA_DIR
from data_loader import load_data

def create_document_term_matrix(text_data):
    """ Convert text data into a document-term matrix. """
    vectorizer = CountVectorizer()
    dtm = vectorizer.fit_transform(text_data)
    return dtm, vectorizer.get_feature_names_out()

if __name__ == "__main__":
    data = load_data('millionsong_dataset.zip')
    dtm, features = create_document_term_matrix(data['text'])