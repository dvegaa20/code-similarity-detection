import pandas as pd
import os
import argparse

def load_qrel(filepath):
    """
    Loads a query relevance (qrel) file and returns a list of code pairs.

    Parameters
    ----------
    filepath : str
        The path to the qrel file to load.

    Returns
    -------
    list of tuples
        A list of tuples, each containing two code identifiers and a default
        similarity label of 1.
    """

    pairs = []
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                id1, id2 = parts
                pairs.append((id1, id2, 1))
    return pairs

def read_source_code(filepath):
    """
    Reads a source code file and returns its contents as a string.

    Parameters
    ----------
    filepath : str
        The path to the source code file to read.

    Returns
    -------
    str
        The source code as a string.
    """
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
        return file.read()

def generate_dataset(language):
    """
    Generates a dataset for a given language, using the query relevance file (qrel) 
    provided by the authors of the FIRE 2014 dataset. This function reads the qrel 
    file, loads the respective source code files and generates a CSV file with the 
    dataset.

    Parameters
    ----------
    language : str
        The language for which to generate the dataset. Supported values are 'java' 
        and 'c'.

    Returns
    -------
    None
    """
    
    BASE_FOLDER = '/Users/diegovega/Developer/Code Similarity Project/fire14-source-code-training-dataset'

    if language.lower() == 'java':
        source_folder = os.path.join(BASE_FOLDER, 'Java')
        qrel_file = os.path.join(BASE_FOLDER, 'SOCO14-java.qrel')
        output_csv = 'data/dataset_java.csv'
    elif language.lower() == 'c':
        source_folder = os.path.join(BASE_FOLDER, 'C')
        qrel_file = os.path.join(BASE_FOLDER, 'SOCO14-c.qrel')
        output_csv = 'data/dataset_c.csv'
    else:
        raise ValueError("Unsupported language. Use 'java' or 'c'.")

    pairs = load_qrel(qrel_file)
    data = []

    for id1, id2, label in pairs:
        file1_path = os.path.join(source_folder, id1)
        file2_path = os.path.join(source_folder, id2)

        if os.path.exists(file1_path) and os.path.exists(file2_path):
            code1 = read_source_code(file1_path)
            code2 = read_source_code(file2_path)
            data.append({
                'code1': code1,
                'code2': code2,
                'similar': label
            })
        else:
            print(f"Warning: Files not found: {id1}, {id2}")

    df = pd.DataFrame(data)
    os.makedirs('data', exist_ok=True)
    df.to_csv(output_csv, index=False, quotechar='"')
    print(f"Dataset successfully generated at {output_csv} with {len(df)} pairs!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset generator for code similarity detection.")
    parser.add_argument('--lang', type=str, required=True)
    args = parser.parse_args()

    generate_dataset(args.lang)
