import torch


def generate_dim_cutoff_embedding(embeds, text_len_tensor, cutoff_ratio=0.2):
    batch_size, max_seq_len, dim = embeds.shape
    cutoff_embeds = embeds.clone()

    for i in range(batch_size):
        cutoff_length = int(dim * cutoff_ratio)
        zero_index = torch.randint(dim, (cutoff_length,))

        for j in range(text_len_tensor[i]):  # Apply cutoff to only the valid length
            cutoff_embeds[i, j, zero_index] = 0

    return cutoff_embeds
