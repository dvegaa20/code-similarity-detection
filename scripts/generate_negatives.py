import pandas as pd
import os
import random
import argparse

def load_positive_dataset(filepath):
    """
    Loads a positive dataset from a CSV file.

    Parameters
    ----------
    filepath : str
        The path to the CSV file containing the positive dataset.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the loaded positive dataset.
    """

    return pd.read_csv(filepath)

def list_source_files(source_folder, extension):
    """
    Lists all source files in the given folder with the given extension.

    Parameters
    ----------
    source_folder : str
        The folder containing the source files to list.
    extension : str
        The file extension to filter the files by.

    Returns
    -------
    list
        A list of file names with the given extension.
    """
    
    files = os.listdir(source_folder)
    return [f for f in files if f.endswith(extension)]

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

def generate_negatives(language):
    """
    Generates a negative dataset for the specified language.

    Parameters
    ----------
    language : str
        The language for which the negative dataset will be generated. Possible
        values are 'java' or 'c'.

    Returns
    -------
    None
    """
    
    BASE_FOLDER = '/Users/diegovega/Developer/Code Similarity Project/fire14-source-code-training-dataset'
    DATA_FOLDER = 'data'

    if language.lower() == 'java':
        source_folder = os.path.join(BASE_FOLDER, 'java')
        positive_dataset_csv = os.path.join(DATA_FOLDER, 'dataset_java.csv')
        output_csv = os.path.join(DATA_FOLDER, 'dataset_java_negatives.csv')
        extension = '.java'
    elif language.lower() == 'c':
        source_folder = os.path.join(BASE_FOLDER, 'c')
        positive_dataset_csv = os.path.join(DATA_FOLDER, 'dataset_c.csv')
        output_csv = os.path.join(DATA_FOLDER, 'dataset_c_negatives.csv')
        extension = '.c'
    else:
        raise ValueError("Lenguaje no soportado. Usa 'java' o 'c'.")

    positives = load_positive_dataset(positive_dataset_csv)

    # List source files
    files = list_source_files(source_folder, extension)
    print(f"Total archivos en {language.upper()}: {len(files)}")

    # Create a set of pairs already used to avoid repetition
    positive_codes = set(zip(positives['code1'], positives['code2']))

    data = []
    attempts = 0
    max_attempts = 10000

    # NNumber of negatives to generate = same as positives
    num_negatives = len(positives)

    while len(data) < num_negatives and attempts < max_attempts:
        file1, file2 = random.sample(files, 2)

        path1 = os.path.join(source_folder, file1)
        path2 = os.path.join(source_folder, file2)

        try:
            code1 = read_source_code(path1)
            code2 = read_source_code(path2)

            # Verify that this pair is not a duplicate or in positives
            if (code1, code2) not in positive_codes and (code2, code1) not in positive_codes and code1 != code2:
                data.append({
                    'code1': code1,
                    'code2': code2,
                    'similar': 0
                })
        except Exception as e:
            print(f"Error reading files: {file1}, {file2} — {e}")

        attempts += 1

    df = pd.DataFrame(data)
    os.makedirs(DATA_FOLDER, exist_ok=True)
    df.to_csv(output_csv, index=False, quotechar='"')
    print(f"Negative dataset successfully generated at {output_csv} with {len(df)} pairs!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Negative dataset generator for code similarity.")
    parser.add_argument('--lang', type=str, required=True)
    args = parser.parse_args()

    generate_negatives(args.lang)
