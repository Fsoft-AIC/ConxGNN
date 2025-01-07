import torch


def feature_packing(multimodal_feature, lengths):
    batch_size = lengths.size(0)
    node_features = []

    for feature in multimodal_feature:  # 3 modalities, 3 features
        for j in range(batch_size):
            # cur_len is sequentially being 36,62,60,57,..... the length of the dialog
            cur_len = lengths[j].item()
            # append the node_feature of the current dialog containing cur_len sentences-nodes
            node_features.append(feature[j, :cur_len])
    node_features = torch.cat(node_features, dim=0)

    return node_features


def multi_concat(nodes_feature, lengths, n_modals):
    sum_length = lengths.sum().item()
    feature = []
    for j in range(n_modals):
        feature.append(nodes_feature[j * sum_length:(j + 1) * sum_length])

    feature = torch.cat(feature, dim=-1)

    return feature
