import torch
import numpy as np
from torch_geometric.data import Batch
from typing import List, Tuple
import torch.nn.functional as F


# These are used in GNN.py
def hetero_to_homo(x_dict, edge_index_dict):
    # Step 1: Concatenate node features
    node_features = torch.cat(
        [x_dict["audio"], x_dict["text"], x_dict["visual"]], dim=0
    )

    # Step 2: Combine edge indices
    # Calculate offsets for each node type
    offsets = {
        "audio": 0,
        "text": x_dict["audio"].size(0),
        "visual": x_dict["audio"].size(0) + x_dict["text"].size(0),
    }

    # Adjust edge indices
    adjusted_edge_indices = []

    for (src, _, dst), edge_index in edge_index_dict.items():
        new_edge_index = edge_index.clone()
        new_edge_index[0, :] += offsets[src]
        new_edge_index[1, :] += offsets[dst]
        adjusted_edge_indices.append(new_edge_index)

    # Combine all edge indices into one tensor
    combined_edge_index = torch.cat(adjusted_edge_indices, dim=1)

    # node_features now contains the concatenated features
    # combined_edge_index contains the combined edge indices
    return (
        node_features,
        combined_edge_index,
        offsets["audio"],
        offsets["text"],
        offsets["visual"],
    )


def hetero_to_homo_keep_edge_types(x_dict, edge_index_dict):
    # Step 1: Concatenate node features
    node_features = torch.cat(
        [x_dict["audio"], x_dict["text"], x_dict["visual"]], dim=0
    )

    # Step 2: Calculate offsets for each node type
    offsets = {
        "audio": 0,
        "text": x_dict["audio"].size(0),
        "visual": x_dict["audio"].size(0) + x_dict["text"].size(0),
    }

    # Step 3: Adjust edge indices and create edge type tensor
    adjusted_edge_indices = []
    edge_types = []

    edge_type_mapping = {
        edge_type: i for i, edge_type in enumerate(edge_index_dict.keys())
    }

    for (src, edge_type, dst), edge_index in edge_index_dict.items():
        new_edge_index = edge_index.clone()
        new_edge_index[0, :] += offsets[src]
        new_edge_index[1, :] += offsets[dst]

        adjusted_edge_indices.append(new_edge_index)
        edge_types.append(
            torch.full(
                (edge_index.size(1),),
                edge_type_mapping[(src, edge_type, dst)],
                dtype=torch.long,
            )
        )

    # Combine all edge indices and edge types into one tensor
    combined_edge_index = torch.cat(adjusted_edge_indices, dim=1)
    combined_edge_types = torch.cat(edge_types)

    return (
        node_features,
        combined_edge_index,
        combined_edge_types,
        offsets["audio"],
        offsets["text"],
        offsets["visual"],
    )


def hetero_to_homo_4mod(x_dict, edge_index_dict):
    # Step 1: Concatenate node features
    node_features = torch.cat(
        [x_dict["audio"], x_dict["text"], x_dict["visual"], x_dict["speakers"]], dim=0
    )

    # Step 2: Combine edge indices
    # Offset for each node type
    offset_audio = 0
    offset_text = x_dict["audio"].size(0)
    offset_visual = x_dict["audio"].size(0) + x_dict["text"].size(0)
    offset_speaker = (
        x_dict["audio"].size(0) + x_dict["text"].size(0) + x_dict["visual"].size(0)
    )

    # Adjust edge indices
    adjusted_edge_indices = []

    for (src, _, dst), edge_index in edge_index_dict.items():
        new_edge_index = edge_index.clone()

        if src == "audio":
            new_edge_index[0, :] += offset_audio
        elif src == "text":
            new_edge_index[0, :] += offset_text
        elif src == "visual":
            new_edge_index[0, :] += offset_visual
        elif src == "speakers":
            new_edge_index[0, :] += offset_speaker

        if dst == "audio":
            new_edge_index[1, :] += offset_audio
        elif dst == "text":
            new_edge_index[1, :] += offset_text
        elif dst == "visual":
            new_edge_index[1, :] += offset_visual
        elif dst == "speakers":
            new_edge_index[1, :] += offset_speaker

        adjusted_edge_indices.append(new_edge_index)

    # Combine all edge indices into one tensor
    combined_edge_index = torch.cat(adjusted_edge_indices, dim=1)

    # node_features now contains the concatenated features
    # combined_edge_index contains the combined edge indices
    return (
        node_features,
        combined_edge_index,
        offset_audio,
        offset_text,
        offset_visual,
        offset_speaker,
    )


def hetero_to_homo_keep_edge_types_4mod(x_dict, edge_index_dict):
    # Step 1: Concatenate node features
    node_features = torch.cat(
        [x_dict["audio"], x_dict["text"], x_dict["visual"], x_dict["speakers"]], dim=0
    )

    # Step 2: Combine edge indices
    # Offset for each node type
    offset_audio = 0
    offset_text = x_dict["audio"].size(0)
    offset_visual = x_dict["audio"].size(0) + x_dict["text"].size(0)
    offset_speaker = (
        x_dict["audio"].size(0) + x_dict["text"].size(0) + x_dict["visual"].size(0)
    )

    # Adjust edge indices and create edge type tensor
    adjusted_edge_indices = []
    edge_types = []

    edge_type_mapping = {
        edge_type: i for i, edge_type in enumerate(edge_index_dict.keys())
    }

    for (src, edge_type, dst), edge_index in edge_index_dict.items():
        new_edge_index = edge_index.clone()

        if src == "audio":
            new_edge_index[0, :] += offset_audio
        elif src == "text":
            new_edge_index[0, :] += offset_text
        elif src == "visual":
            new_edge_index[0, :] += offset_visual
        elif src == "speakers":
            new_edge_index[0, :] += offset_speaker

        if dst == "audio":
            new_edge_index[1, :] += offset_audio
        elif dst == "text":
            new_edge_index[1, :] += offset_text
        elif dst == "visual":
            new_edge_index[1, :] += offset_visual
        elif dst == "speakers":
            new_edge_index[1, :] += offset_speaker

        adjusted_edge_indices.append(new_edge_index)
        edge_types.append(
            torch.full(
                (edge_index.size(1),),
                edge_type_mapping[(src, edge_type, dst)],
                dtype=torch.long,
            )
        )

    # Combine all edge indices and edge types into one tensor
    combined_edge_index = torch.cat(adjusted_edge_indices, dim=1)
    combined_edge_types = torch.cat(edge_types)

    # node_features now contains the concatenated features
    # combined_edge_index contains the combined edge indices
    # combined_edge_types contains the edge types
    return (
        node_features,
        combined_edge_index,
        combined_edge_types,
        offset_audio,
        offset_text,
        offset_visual,
        offset_speaker,
    )


######################################
# These are used in HeteroGraphModel #
######################################
def edge_perms(wp: int, wf: int, length: int, mode: str):
    """
    mode can be either: cross, temp_past, temp_future, temp_self
    """
    perms = set()
    if mode == "cross":
        for node_index in range(length):
            perms.add(torch.tensor([node_index, node_index]))
        result = torch.stack(list(perms)).t().contiguous()
        return result
    elif mode == "temp_past":
        if length == 1:
            return torch.tensor([[0], [0]])
        array = np.arange(length)
        for node_index in range(length):
            # since wf=5 and wp=11, it means connect upto 5-1=4 nodes
            # (-1 because in python the upper bound is excluded) in the future
            # and upto 11 nodes into the past
            eff_array = array[max(0, node_index - wp): min(length, node_index)]
            for item in eff_array:
                perms.add(torch.tensor([node_index, item]))
        result = torch.stack(list(perms)).t().contiguous()
        return result
    elif mode == "temp_future":
        if length == 1:
            return torch.tensor([[0], [0]])
        array = np.arange(length)
        for node_index in range(length):
            # since wf=5 and wp=11, it means connect upto 5-1=4 nodes
            # (-1 because in python the upper bound is excluded) in the future
            # and upto 11 nodes into the past
            eff_array = array[max(0, node_index + 1): min(length, node_index + wf)]
            for item in eff_array:
                perms.add(torch.tensor([node_index, item]))
        result = torch.stack(list(perms)).t().contiguous()
        return result
    elif mode == "temp_self":
        for node_index in range(length):
            perms.add(torch.tensor([node_index, node_index]))
        result = torch.stack(list(perms)).t().contiguous()
        return result


def create_branch(
    batch_hetero_data: Batch, window: Tuple[int, int], use_speaker: str
) -> Batch:
    """
    Args:
        window: (wp, wf) for past window and future window
        batch_hetero_data (Batch)

    Returns:
        batch_hetero_data: it have edge index and edge type info for each hetero_data
    """
    list_hetero_data = (
        batch_hetero_data.to_data_list()
    )  # this list will be a separated object
    for idx, hetero_data in enumerate(list_hetero_data):
        if use_speaker in ["no", "add"]:
            length = hetero_data["speakers"].num_nodes
            for relation in ["past", "future", "self"]:
                for modality in ["audio", "visual", "text"]:
                    hetero_data[modality, relation, modality].edge_index = edge_perms(
                        wp=window[0],
                        wf=window[1],
                        length=length,
                        mode=f"temp_{relation}",
                    )

            modality_pairs = [
                ("audio", "visual"),
                ("audio", "text"),
                ("visual", "audio"),
                ("visual", "text"),
                ("text", "audio"),
                ("text", "visual"),
            ]
            for src, dst in modality_pairs:
                hetero_data[src, "cross", dst].edge_index = edge_perms(
                    wp=window[0], wf=window[1], length=length, mode="cross"
                )
        elif use_speaker == "separate":
            length = hetero_data["speakers"].num_nodes

            # Temporal edges
            for edge_type in ["past", "future", "self"]:
                for modality in ["audio", "visual", "text", "speakers"]:
                    hetero_data[modality, edge_type, modality].edge_index = edge_perms(
                        wp=window[0],
                        wf=window[1],
                        length=length,
                        mode=f"temp_{edge_type}",
                    )

            # Cross modality edges
            modality_pairs = [
                ("audio", "visual"),
                ("audio", "text"),
                ("audio", "speakers"),
                ("visual", "audio"),
                ("visual", "text"),
                ("visual", "speakers"),
                ("text", "audio"),
                ("text", "visual"),
                ("text", "speakers"),
                ("speakers", "audio"),
                ("speakers", "visual"),
                ("speakers", "text"),
            ]
            for src, dst in modality_pairs:
                hetero_data[src, "cross", dst].edge_index = edge_perms(
                    wp=window[0], wf=window[1], length=length, mode="cross"
                )

    new_batch_hetero_data = Batch.from_data_list(list_hetero_data)
    return new_batch_hetero_data.edge_index_dict


# Function to compute similarity matrix
def compute_similarity_matrix(tensor1, tensor2, gamma=1.0):
    # normalize
    tensor1_norm = F.normalize(tensor1, p=2, dim=1)
    tensor2_norm = F.normalize(tensor2, p=2, dim=1)
    # Compute cosine similarity
    sim_matrix = torch.mm(tensor1_norm, tensor2_norm.t())
    # Calculate arccos of the similarity and apply the formulas
    arc_cos_sim = torch.arccos(sim_matrix.clamp(-1 + 1e-7, 1 - 1e-7)) / torch.pi
    weight_matrix = gamma * (1 - arc_cos_sim)
    return weight_matrix


def is_symmetric(tensor):  # Check if the tensor is equal to its transpose
    return torch.equal(tensor, tensor.T)


def get_edge_weight(x_dict, edge_index_dict, gamma=1.0, use_speaker=False):
    a = x_dict["audio"]
    v = x_dict["visual"]
    t = x_dict["text"]

    sim_a_a = compute_similarity_matrix(a, a, gamma)
    sim_t_t = compute_similarity_matrix(t, t, gamma)
    sim_v_v = compute_similarity_matrix(v, v, gamma)

    cross_sim_a_t = compute_similarity_matrix(a, t, gamma)
    cross_sim_a_v = compute_similarity_matrix(a, v, gamma)

    cross_sim_v_a = compute_similarity_matrix(v, a, gamma)
    cross_sim_v_t = compute_similarity_matrix(v, t, gamma)

    cross_sim_t_a = compute_similarity_matrix(t, a, gamma)
    cross_sim_t_v = compute_similarity_matrix(t, v, gamma)
    edge_weight_dict = {}

    for key, edge_index in edge_index_dict.items():
        src_nodes = edge_index[0]  # Source nodes
        dst_nodes = edge_index[1]  # Destination nodes

        if key[0] == "audio" and key[2] == "audio":
            similarity_matrix = sim_a_a
        elif key[0] == "visual" and key[2] == "visual":
            similarity_matrix = sim_v_v
        elif key[0] == "text" and key[2] == "text":
            similarity_matrix = sim_t_t

        elif key[0] == "audio" and key[2] == "visual":
            similarity_matrix = cross_sim_a_v
        elif key[0] == "audio" and key[2] == "text":
            similarity_matrix = cross_sim_a_t
        elif key[0] == "visual" and key[2] == "audio":
            similarity_matrix = cross_sim_v_a
        elif key[0] == "visual" and key[2] == "text":
            similarity_matrix = cross_sim_v_t
        elif key[0] == "text" and key[2] == "audio":
            similarity_matrix = cross_sim_t_a
        elif key[0] == "text" and key[2] == "visual":
            similarity_matrix = cross_sim_t_v

        # Get the edge weights by indexing into the similarity matrix
        edge_weights = similarity_matrix[src_nodes, dst_nodes]
        # Store in the edge_weight_dict
        edge_weight_dict[key] = edge_weights
    return edge_weight_dict


def process_wp_wf(wp_wf) -> List:
    try:
        branches = []
        current_branch = []
        detail_branches = []
        for item in wp_wf:
            if item == 0:
                assert (
                    len(current_branch) % 2 == 0
                ), "Branch must contain an even no. elements."
                branches.append(current_branch)
                current_branch = []
            else:
                current_branch.append(item)
        # check last branch
        if len(current_branch) > 0:
            assert (
                len(current_branch) % 2 == 0
            ), "Branch must contain an even no. elements."
            branches.append(current_branch)
        for branch in branches:
            windows = [(branch[i], branch[i + 1]) for i in range(0, len(branch), 2)]
            detail_branches.append(windows)
        return detail_branches
    except AssertionError as e:
        print(f"Error: {e}")
        return False


#####################################
# These are used in HyperGraphModel #
#####################################
def create_hyper_index(text_len_tensor, use_speaker: str, device):
    if use_speaker in ["no", "add"]:
        num_modality = 3
    elif use_speaker == "separate":
        num_modality = 4
    dialog_offset = 0
    batch_unimodal_len = int(torch.sum(text_len_tensor))
    edge_count = 0
    index1 = []
    index2 = []
    hyperedge_type1 = []
    for len in text_len_tensor:
        # unimodal edges :
        # unimodal audio edges
        nodes_a = list(range(len))
        nodes_a = [j + 0 for j in nodes_a]
        nodes_a = [j + dialog_offset for j in nodes_a]
        # unimodal text edges
        nodes_t = list(range(len))
        nodes_t = [j + batch_unimodal_len for j in nodes_t]
        nodes_t = [j + dialog_offset for j in nodes_t]
        # unimodal visual edges
        nodes_v = list(range(len))
        nodes_v = [j + batch_unimodal_len * 2 for j in nodes_v]
        nodes_v = [j + dialog_offset for j in nodes_v]
        if use_speaker in ["no", "add"]:
            index1 = index1 + nodes_a + nodes_t + nodes_v
        elif use_speaker in ["separate"]:
            # unimodal visual edges
            nodes_s = list(range(len))
            nodes_s = [j + batch_unimodal_len * 3 for j in nodes_s]
            nodes_s = [j + dialog_offset for j in nodes_s]
            index1 = index1 + nodes_a + nodes_t + nodes_v + nodes_s
        for _ in range(len):
            if use_speaker in ["no", "add"]:
                index1 = index1 + [nodes_a[_]] + [nodes_t[_]] + [nodes_v[_]]
            elif use_speaker in ["separate"]:
                index1 = (
                    index1 + [nodes_a[_]] + [nodes_t[_]] + [nodes_v[_]] + [nodes_s[_]]
                )
        for _ in range(len + num_modality):
            if _ < num_modality:
                index2 = index2 + [edge_count] * len
                hyperedge_type1 = hyperedge_type1 + [0]
            else:
                index2 = index2 + [edge_count] * num_modality
                hyperedge_type1 = hyperedge_type1 + [1]
            edge_count = edge_count + 1
        dialog_offset += len
    index1 = torch.LongTensor(index1).view(1, -1)
    index2 = torch.LongTensor(index2).view(1, -1)
    hyperedge_index = torch.cat([index1, index2], dim=0).to(device)
    hyperedge_type1 = torch.LongTensor(hyperedge_type1).view(-1, 1).to(device)
    return hyperedge_index, hyperedge_type1
