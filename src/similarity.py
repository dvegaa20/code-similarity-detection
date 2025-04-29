from torch.nn.functional import cosine_similarity


def calculate_similarity(emb1, emb2):
    """
    Calculates the cosine similarity between two embeddings.

    Parameters
    ----------
    emb1 : torch.Tensor
        The first embedding.
    emb2 : torch.Tensor
        The second embedding.

    Returns
    -------
    float
        The cosine similarity between the two embeddings.
    """

    return cosine_similarity(emb1, emb2, dim=0).item()
