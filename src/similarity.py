from torch.nn.functional import cosine_similarity

def calculate_similarity(emb1, emb2):
    """
    Calculates the cosine similarity between two code embeddings.
    """

    return cosine_similarity(emb1, emb2, dim=0).item()
