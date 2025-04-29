import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def load_model():
    """
    Loads the pre-trained UnixCoder model and tokenizer, and moves the model to the
    GPU if available.

    Returns
    -------
    tokenizer : transformers.AutoTokenizer
        The pre-trained tokenizer.
    model : transformers.AutoModel
        The pre-trained model.
    device : torch.device
        The device where the model is loaded.
    """

    tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
    model = AutoModel.from_pretrained("microsoft/unixcoder-base")
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return tokenizer, model, device


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


def vectorize_corpus(corpus, tokenizer, model, device, batch_size=16):
    """
    Vectorizes a corpus of text using a transformer model.

    Parameters
    ----------
    corpus : list of str
        The corpus of text to vectorize.
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer to use for tokenizing the corpus.
    model : transformers.PreTrainedModel
        The transformer model to use for encoding the corpus.
    device : torch.device
        The device to use for encoding the corpus.
    batch_size : int, optional
        The batch size to use when encoding the corpus. Defaults to 16.

    Returns
    -------
    torch.Tensor
        The vectorized corpus, with shape (len(corpus), hidden_size).
    """

    embeddings = []

    for i in tqdm(
        range(0, len(corpus), batch_size), desc="Encoding corpus with batching"
    ):
        batch = corpus[i : i + batch_size]
        with torch.no_grad():
            inputs = tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)

            outputs = model(**inputs)
            pooled_embeddings = mean_pooling(outputs, inputs["attention_mask"])

            embeddings.append(pooled_embeddings)

    embeddings = torch.cat(embeddings, dim=0)
    return embeddings
