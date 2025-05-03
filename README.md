# 📘 Code Similarity Detection Project

## 📄 Project Description

This project aims to develop a tool for detecting similarity between code fragments, focusing on identifying potential cases of plagiarism or code reuse.
Text processing and machine learning techniques are applied to solve the problem as a binary classification task.

The project uses code datasets in Java and C, based on the SOCO dataset (FIRE 2014).

## 🏗️ Project Structure

```
📂 code-similarity-detection
├── data/
│ ├── dataset_c_full.csv
│ ├── dataset_java_full.csv
│
├── logs/
│ ├── flaml_c.log
│ ├── flaml_java.log
|
├── scripts/
│ ├── combine_datasets.py
│ ├── generate_dataset.py
│ ├── generate_negatives.py
│
├── src/
│ ├── model.py
│ ├── preprocessing.py
│ ├── similarity.py
│
├── .gitignore
├── main.py
├── README.md
├── requirements.txt

```

## 📦 Install Dependencies

Before running the project, install the required dependencies using the `requirements.txt` file:

```
pip install -r requirements.txt
```

(This will automatically install pandas and scikit-learn, which are required for data processing and model building.)

## 🛠️ Usage, run the project

### 📊 1. Dataset Preparation

Make sure you have the original SOCO dataset structure like this:

```
📂 fire14-source-code-training-dataset/
├── c/
├── java/
├── SOCO14-c.qrel
├── SOCO14-java.qrel
```

### ✨ 2. Generate Positive (Plagiarized) Pairs

Run the following commands to generate positive pairs from .qrel files:

- For Java:

```
python scripts/generate_dataset.py --lang java
```

- For C:

```
python scripts/generate_dataset.py --lang c
```

In the `data/` directory output files will be saved as:

- `data/dataset_c.csv`
- `data/dataset_java.csv`

(These contain only plagiarized/similar code pairs with label 1.)

### ❌ 3. Generate Negative (Non-Plagiarized) Pairs

Run these commands to generate random non-plagiarized pairs:

- For Java:

```
python scripts/generate_negatives.py --lang java
```

- For C:

```
python scripts/generate_negatives.py --lang c
```

In the `data/` directory output files will be saved as:

- `data/dataset_c_negatives.csv`
- `data/dataset_java_negatives.csv`

(These contain only non-plagiarized code pairs with label 0.)

### 📦 4. Combine Datasets

After generating positives and negatives, combine them into full datasets:

- For Java:

```
python scripts/combine_datasets.py --lang java
```

- For C:

```
python scripts/combine_datasets.py --lang c
```

In the `data/` directory output files will be saved as:

- `data/dataset_c_full.csv`: Contains **26 plagiarized pairs** and **26 non-plagiarized pairs**.
- `data/dataset_java_full.csv`: Contains **84 plagiarized pairs** and **84 non-plagiarized pairs**.

(These contain both plagiarized and non-plagiarized code pairs with labels 0 and 1.)

You can safely delete the intermediate files `dataset_c.csv`, `dataset_java.csv`, `dataset_c_negatives.csv`, and `dataset_java_negatives.csv` if you want to clean up.

### 🏋🏽🔍 5. Train and Evaluate the Model

Run the main script to train a classifier and evaluate its performance:

- For Java:

```
python main.py --lang java
```

- For C:

```
python main.py --lang c
```

falta concluir
