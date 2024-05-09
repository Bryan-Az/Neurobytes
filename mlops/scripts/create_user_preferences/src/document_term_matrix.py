import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from config import DATA_DIR
from data_loader import load_data

def create_document_term_matrix(data):
    """ Convert text data into a document-term matrix. """
    text_corpus = create_text_corpus(data)
    vectorizer = CountVectorizer()
    dtm = vectorizer.fit_transform(text_corpus)
    return dtm, vectorizer.get_feature_names_out()

from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import nltk
nltk.download('punkt')
nltk.download('stopwords')
def create_text_corpus(data, lyric_chars=1000):
    """
    Prepares the text data before input to the document-term matrix.
    It incorporates artist name and a subset of lyrics.
    """

    def top_n_words(text, n=5):
        '''
        Applied on the lyrics to shrink weight of lyrics on similarity
        '''
        freq_dist = FreqDist(word_tokenize(text))
        # remove filler words
        freq_dist = FreqDist({key: val for key, val in freq_dist.items() if key not in nltk.corpus.stopwords.words('english')})
        # select only the words from the (word, freq) tuple list
        words = ' '.join([word for word, freq in freq_dist.most_common()])
        return words
    
    artist_names = data['artist']
    lyrics = data['text']
    top_words = lyrics.apply(lambda x: top_n_words(x)) 
    
    return artist_names + ' ' + top_words


if __name__ == "__main__":
    data = load_data('millionsong_dataset.zip')
    dtm, features = create_document_term_matrix(data)