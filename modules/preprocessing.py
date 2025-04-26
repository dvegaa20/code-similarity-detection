import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def load_dataset(filepath):
    """
    Loads a dataset from the given CSV file.

    Parameters
    ----------
    filepath : str
        The path to the CSV file to load.

    Returns
    -------
    df : pd.DataFrame
        A pandas DataFrame containing the loaded dataset.
    """
    
    return pd.read_csv(filepath, quotechar='"')

def vectorize_corpus(corpus):
    """
    Vectorizes the given corpus.

    Parameters
    ----------
    corpus : array-like
        The corpus to vectorize.

    Returns
    -------
    vectorizer : TfidfVectorizer
        The object that was used to vectorize the corpus.
    X : array-like
        The transformed corpus.
    """
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    return vectorizer, X
