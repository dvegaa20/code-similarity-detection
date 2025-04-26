from modules.preprocessing import load_dataset, vectorize_corpus
from modules.similarity import calculate_similarity
from modules.model import train_model, evaluate_model
import pandas as pd

# Cargar dataset
df = load_dataset('data/dataset.csv')

# Preprocesar corpus
corpus = df['codigo1'] + df['codigo2']
vectorizer, X_vectorized = vectorize_corpus(corpus)

# Calcular similitudes
df['similitud'] = [calculate_similarity(vectorizer, row['codigo1'], row['codigo2']) for _, row in df.iterrows()]

# Preparar datos para el modelo
X = df[['similitud']]
y = df['similar']

# Entrenar y evaluar
model, X_test, y_test = train_model(X, y)
report = evaluate_model(model, X_test, y_test)

print(report)
