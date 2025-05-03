import argparse
from src.preprocessing import load_dataset, load_model, vectorize_corpus
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

    if language.lower() == "java":
        dataset_path = "data/dataset_java_full.csv"
    elif language.lower() == "c":
        dataset_path = "data/dataset_c_full.csv"
    else:
        raise ValueError("Unsupported language. Use 'java' or 'c'.")

    df = load_dataset(dataset_path)
    tokenizer, model, device = load_model()

    code1_embeddings = vectorize_corpus(df["code1"].to_list(), tokenizer, model, device)
    code2_embeddings = vectorize_corpus(df["code2"].to_list(), tokenizer, model, device)

    df["similarity"] = [
        calculate_similarity(code1_embeddings[idx], code2_embeddings[idx])
        for idx in range(len(df))
    ]

    X = df[["similarity"]]
    y = df["similar"]

    model, X_test, y_test = train_model(X, y)
    report = evaluate_model(model, X_test, y_test)

    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code similarity detection system.")
    parser.add_argument(
        "--lang", type=str, required=True, help="Lenguaje: 'java' o 'c'"
    )
    args = parser.parse_args()

    main(args.lang)
