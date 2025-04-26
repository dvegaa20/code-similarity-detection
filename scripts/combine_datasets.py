import pandas as pd
import argparse
import os

def combine_datasets(language):
    """
    Combine positive and negative datasets into a full dataset.

    Parameters
    ----------
    language : str
        The language of the dataset. Supported values are 'java' and 'c'.

    Notes
    -----
    This script assumes that the positive and negative datasets are stored in the 'data' folder
    as 'dataset_<language>.csv' and 'dataset_<language>_negatives.csv' respectively. The output
    file is named 'dataset_<language>_full.csv'.

    The script concatenates the two datasets, shuffles the combined dataset using the random
    seed 42, and writes the output to the specified file.
    """
    
    DATA_FOLDER = 'data'

    if language.lower() == 'java':
        positive_csv = os.path.join(DATA_FOLDER, 'dataset_java.csv')
        negative_csv = os.path.join(DATA_FOLDER, 'dataset_java_negatives.csv')
        output_csv = os.path.join(DATA_FOLDER, 'dataset_java_full.csv')
    elif language.lower() == 'c':
        positive_csv = os.path.join(DATA_FOLDER, 'dataset_c.csv')
        negative_csv = os.path.join(DATA_FOLDER, 'dataset_c_negatives.csv')
        output_csv = os.path.join(DATA_FOLDER, 'dataset_c_full.csv')
    else:
        raise ValueError("Unsupported language. Use 'java' or 'c'.")

    positives = pd.read_csv(positive_csv)
    negatives = pd.read_csv(negative_csv)

    # Combine and shuffle
    full_dataset = pd.concat([positives, negatives]).sample(frac=1, random_state=42).reset_index(drop=True)

    full_dataset.to_csv(output_csv, index=False, quotechar='"')
    print(f"Full dataset successfully generated at {output_csv} with {len(full_dataset)} examples!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine positive and negative examples into a complete dataset.")
    parser.add_argument('--lang', type=str, required=True)
    args = parser.parse_args()

    combine_datasets(args.lang)
