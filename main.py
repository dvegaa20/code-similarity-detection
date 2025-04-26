import pandas as pd
import argparse
from src.preprocessing import load_dataset, vectorize_corpus
from src.similarity import calculate_similarity
from src.model import train_model, evaluate_model

def main(language):
    """
    Main entry point of the program.

    Parameters
    ----------
    language : str
        The programming language of the dataset to load. Supported values are 'java' and 'c'.

    Notes
    -----
    The program loads the specified dataset, vectorizes the corpus, calculates the similarity between each pair of
    codes and trains a model to predict the similarity. The model is then evaluated and the results are printed to the
    console.
    """
    
    if language.lower() == 'java':
        dataset_path = 'data/dataset_java_full.csv'
    elif language.lower() == 'c':
        dataset_path = 'data/dataset_c_full.csv'
    else:
        raise ValueError("Lenguaje no soportado. Usa 'java' o 'c'.")

    df = load_dataset(dataset_path)

    corpus = df['code1'] + df['code2']
    vectorizer, _ = vectorize_corpus(corpus)

    # Calculate similarities
    df['similarity'] = [calculate_similarity(vectorizer, row['code1'], row['code2']) for _, row in df.iterrows()]

    # Prepare data for the model
    X = df[['similarity']]
    y = df['similar']

    model, X_test, y_test = train_model(X, y)
    report = evaluate_model(model, X_test, y_test)

    print(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sistema de detección de similitud de código.")
    parser.add_argument('--lang', type=str, required=True, help="Lenguaje: 'java' o 'c'")
    args = parser.parse_args()

    main(args.lang)
