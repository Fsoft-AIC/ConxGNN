import torch
import torch.nn as nn
from .GNNLayer.GNN import get_graph_block
from .utils import process_wp_wf, create_branch, get_edge_weight
from src.TensorGraph import TensorGraph


class HeteroGraphModel(nn.Module):
    def __init__(self, args):
        super(HeteroGraphModel, self).__init__()
        # the number of gnn layers in each gnn block
        gnn_block_num_layers = args.gnn_block_num_layers
        self.gnn_block_trans_heads = args.gnn_block_trans_heads
        self.gnn_end_block_trans_heads = args.gnn_end_block_trans_heads
        # Pair norm
        self.use_pair_norm = args.use_pair_norm
        self.norm_mode = args.norm_mode
        self.norm_scale = args.norm_scale

        # use MMGCN edge weighting for hetero graph model
        self.use_edge_weight_hetero = args.use_edge_weight_hetero

        hidden_channels = args.hidden_size
        drop_p = args.drop_rate
        device = args.device
        self.use_residual = args.use_residual
        self.use_speaker = args.use_speaker

        # it looks like this [11, 9, 8, 7, 5, 5, 3, |, 11, 9, 8, 7, 5, 5, 3, |,11, 9, 8, 7, 5, 5, 3]
        # | separates the branches
        # each couple is a (wp and wf), forming a "layer"
        # each layer is having different wp and wf
        # multiple layer consecutively forming a branch
        wp_wf = args.wp_wf

        # Verify if the the wp_wf are created correctly
        self.branches = process_wp_wf(wp_wf)
        for idx_branch, windows in enumerate(self.branches):
            for idx_window, window in enumerate(windows):  # each window is a block
                if idx_window != (len(windows) - 1):
                    setattr(
                        self,
                        f"br_{str(idx_branch)}_win_{str(idx_window)}",
                        get_graph_block(
                            hidden_channels,
                            device,
                            gnn_block_num_layers,
                            self.gnn_block_trans_heads,
                            self.use_speaker,
                            False,
                            args.gnn_model,
                            drop_p,
                            self.use_pair_norm,
                            self.norm_mode,
                            self.norm_scale,
                        ).to(device),
                    )
                else:
                    setattr(
                        self,
                        f"br_{str(idx_branch)}_win_{str(idx_window)}",
                        get_graph_block(
                            hidden_channels,
                            device,
                            gnn_block_num_layers,
                            self.gnn_end_block_trans_heads,
                            self.use_speaker,
                            True,
                            args.gnn_model,
                            drop_p,
                            self.use_pair_norm,
                            self.norm_mode,
                            self.norm_scale,
                        ).to(device),
                    )
        #################################
        # Concerning which model to use #
        #################################

    def forward(self, ori_tensor_graph) -> TensorGraph:
        # Initialize lists to store the outputs for each modality
        audio_outputs = []
        visual_outputs = []
        text_outputs = []
        if self.use_speaker == "separate":
            speaker_outputs = []
        # Assuming self.windows is a list of window sizes or identifiers
        for idx_branch, branch in enumerate(self.branches):
            batch_hetero_data = ori_tensor_graph.data
            x_dict = batch_hetero_data.x_dict  # this will change after each block
            for idx_window, window in enumerate(branch):
                # create a new edge_index_dict for this
                edge_index_dict = create_branch(
                    batch_hetero_data, window, self.use_speaker
                )

                # Store the original x_dict before applying the GNN
                x_dict_residual = {key: x.clone() for key, x in x_dict.items()}

                # Apply the GNN model for this window
                gnn = getattr(self, f"br_{str(idx_branch)}_win_{str(idx_window)}")
                if self.use_edge_weight_hetero:
                    edge_weight_dict = get_edge_weight(x_dict, edge_index_dict)
                    x_dict = gnn(x_dict, edge_index_dict, edge_weight_dict)
                else:
                    x_dict = gnn(x_dict, edge_index_dict)

                if self.use_residual:
                    # Add the residual connection
                    if idx_window != (len(branch) - 1):
                        trans_head = self.gnn_block_trans_heads
                    else:
                        trans_head = self.gnn_end_block_trans_heads
                    x_dict_residual = {
                        key: x_dict_residual[key].repeat(1, trans_head)
                        if x.size(1) != x_dict_residual[key].size(1)
                        else x_dict_residual[key]
                        for key, x in x_dict.items()
                    }
                    x_dict = {
                        key: x + x_dict_residual[key] for key, x in x_dict.items()
                    }

            # Append the outputs to the respective lists
            audio_outputs.append(x_dict["audio"])
            visual_outputs.append(x_dict["visual"])
            text_outputs.append(x_dict["text"])
            if self.use_speaker == "separate":
                speaker_outputs.append(x_dict["speakers"])

        # Average the tensors for each modality
        average_audio = sum(audio_outputs) / len(audio_outputs)
        average_visual = sum(visual_outputs) / len(visual_outputs)
        average_text = sum(text_outputs) / len(text_outputs)
        if self.use_speaker == "separate":
            average_speaker = sum(speaker_outputs) / len(speaker_outputs)

        result_tensor_graph = ori_tensor_graph.clone()
        # Concatenate the averaged tensors
        if self.use_speaker in ["no", "add"]:
            concatenated_tensor = torch.cat(
                [average_audio, average_text, average_visual], dim=0
            )
            result_tensor_graph.representation = "node_concat_3mod"
            result_tensor_graph.data = concatenated_tensor
        elif self.use_speaker == "separate":
            concatenated_tensor = torch.cat(
                [average_audio, average_text, average_visual, average_speaker], dim=0
            )
            result_tensor_graph.representation = "node_concat_4mod"
            result_tensor_graph.data = concatenated_tensor
        return result_tensor_graph
