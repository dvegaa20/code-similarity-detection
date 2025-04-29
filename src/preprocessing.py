import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
model = AutoModel.from_pretrained("microsoft/unixcoder-base")
model.eval()

def load_dataset(filepath):
    return pd.read_csv(filepath, quotechar='"')

def mean_pooling(model_output, attention_mask):
    """
    Mean Pooling 
    """
    token_embeddings = model_output.last_hidden_state  # (batch_size, sequence_length, hidden_size)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask  # (batch_size, hidden_size)

def vectorize_corpus(corpus):
    """
    Encodes each code snippet separately using mean pooling.
    """
    embeddings = []

    for code_snippet in tqdm(corpus, desc="Encoding corpus with mean pooling"):
        with torch.no_grad():
            inputs = tokenizer(
                code_snippet,
                padding="max_length",
                truncation=True,
                max_length=512,     # UnixCoder expects <= 512 tokens
                return_tensors="pt"
            )
            outputs = model(**inputs)
            pooled_embedding = mean_pooling(outputs, inputs['attention_mask'])  
            embeddings.append(pooled_embedding.squeeze(0))  # (768,) vector

    embeddings = torch.stack(embeddings)  # Shape: (num_samples, embedding_dim)
    return embeddings