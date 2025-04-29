import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
model = AutoModel.from_pretrained("microsoft/unixcoder-base")
model.eval()


def load_dataset(filepath):
    """
    Loads a dataset from a CSV file.

    Parameters
    ----------
    filepath : str
        The path to the CSV file to load.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the loaded dataset.
    """

    return pd.read_csv(filepath, quotechar='"')


def mean_pooling(model_output, attention_mask):
    """
    Applies mean pooling to the token embeddings.

    Parameters
    ----------
    model_output : object
        The output from the transformer model, containing the last hidden state.
    attention_mask : torch.Tensor
        The attention mask indicating which tokens are valid for the input sequence.

    Returns
    -------
    torch.Tensor
        The mean-pooled embeddings for the input sequence.
    """

    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


def vectorize_corpus(corpus):
    """
    Vectorizes a corpus of code snippets by encoding each snippet into a fixed-size embedding.

    Parameters
    ----------
    corpus : list of str
        A list of code snippets to be encoded.

    Returns
    -------
    torch.Tensor
        A tensor containing the embeddings for the entire corpus, where each
        embedding corresponds to a code snippet.
    """

    embeddings = []

    for code_snippet in tqdm(corpus, desc="Encoding corpus with mean pooling"):
        with torch.no_grad():
            inputs = tokenizer(
                code_snippet,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            outputs = model(**inputs)
            pooled_embedding = mean_pooling(outputs, inputs["attention_mask"])
            embeddings.append(pooled_embedding.squeeze(0))

    embeddings = torch.stack(embeddings)
    return embeddings
