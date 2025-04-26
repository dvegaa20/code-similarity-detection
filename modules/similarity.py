from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(vectorizer, code1, code2):
    """
    Calcula la similitud entre dos codigos.

    Parameters
    ----------
    vectorizer : TfidfVectorizer
        El objeto que se utiliza para vectorizar el corpus.
    code1 : str
        El primer codigo.
    code2 : str
        El segundo codigo.

    Returns
    -------
    float
        La similitud entre los dos codigos.
    """

    vec1 = vectorizer.transform([code1])
    vec2 = vectorizer.transform([code2])
    return cosine_similarity(vec1, vec2)[0][0]
