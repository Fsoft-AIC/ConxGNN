import torch
from .InfoNCE import InfoNCE


def compute_infonce_loss(
    orig_output,
    cutoff_output,
    text_len_tensor,
    temperature=0.1,
    negative_mode="paired",
    wp=None,
    wf=None,
):
    batch_size = orig_output.shape[0]
    # Iterate over each dialog in the batch
    total_loss = 0.0
    for i in range(batch_size):
        dialog_length = text_len_tensor[i]
        query_feat = orig_output[i, :dialog_length, :]  # text_len, dim
        positive_feat = cutoff_output[i, :dialog_length, :]  # text_len, dim
        negatives_feat = []
        # Iterate over each sentence in the current dialog
        for j in range(dialog_length):
            # Exclude the current sentence's feature
            if wp is None and wf is None:
                negative_feat = torch.cat(
                    [query_feat[:j], query_feat[j + 1:]]
                )  # (text_len-1,dim)
            elif wp is not None and wf is not None:
                negative_feat = torch.cat(
                    [
                        query_feat[max(0, j - wp): j],
                        query_feat[j + 1: min(dialog_length, j + 1 + wf)],
                    ]
                )  # (text_len - 1, dim)
            negatives_feat.append(negative_feat)
        negatives_feat = torch.stack(negatives_feat, dim=0)
        loss = InfoNCE(temperature=temperature, negative_mode=negative_mode)
        output = loss(query_feat, positive_feat, negatives_feat)
        total_loss += output
    total_loss = total_loss / batch_size
    return total_loss
