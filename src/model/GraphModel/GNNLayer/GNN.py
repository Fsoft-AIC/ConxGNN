import torch.nn as nn
from ..utils import hetero_to_homo, hetero_to_homo_4mod
from torch_geometric.nn import HeteroConv, TransformerConv
from torch_geometric.nn import GraphConv
from .FAConv import FAConv
from .high_fre_conv import highConv
from ..NormLayer import PairNorm
import torch


def get_graph_block(
    hidden_channels,
    device,
    num_layers,
    transformer_heads,
    use_speaker,
    is_end_block,
    gnn_layer,
    drop_p,
    use_pair_norm,
    norm_mode,
    norm_scale,
):
    model_map = {
        "GraphConv": GraphConvHeteroBlock,
        "FAConv": FAConvHeteroBlock,
        "highConv": highConvHeteroBlock,
    }
    if is_end_block:
        concat_head = True
    else:
        concat_head = False
    block = model_map[gnn_layer](
        hidden_channels,
        device,
        num_layers,
        transformer_heads,
        use_speaker,
        concat_head,
        drop_p,
        use_pair_norm,
        norm_mode,
        norm_scale,
    )
    return block


class GraphConvHeteroBlock(nn.Module):
    def __init__(
        self,
        hidden_channels,
        device,
        num_layers,
        transformer_heads,
        use_speaker,
        concat_head,
        drop_p,
        use_pair_norm,
        norm_mode,
        norm_scale,
    ):
        super(GraphConvHeteroBlock, self).__init__()
        self.device = device
        self.use_speaker = use_speaker
        self.convs_1 = nn.ModuleList()
        self.drop_p = drop_p
        for _ in range(num_layers):
            if use_speaker in ["add", "no"]:
                convs = HeteroConv(
                    {
                        ("audio", "past", "audio"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("visual", "past", "visual"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("text", "past", "text"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("audio", "future", "audio"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("visual", "future", "visual"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("text", "future", "text"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("audio", "self", "audio"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("visual", "self", "visual"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("text", "self", "text"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("audio", "cross", "visual"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("audio", "cross", "text"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("visual", "cross", "audio"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("visual", "cross", "text"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("text", "cross", "audio"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("text", "cross", "visual"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                    },
                    aggr="mean",
                )
            elif use_speaker == "separate":
                convs = HeteroConv(
                    {
                        ("audio", "past", "audio"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("visual", "past", "visual"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("text", "past", "text"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("speakers", "past", "speakers"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("audio", "future", "audio"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("visual", "future", "visual"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("text", "future", "text"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("speakers", "future", "speakers"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("audio", "self", "audio"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("visual", "self", "visual"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("text", "self", "text"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("speakers", "self", "speakers"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("audio", "cross", "visual"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("audio", "cross", "text"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("audio", "cross", "speakers"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("visual", "cross", "audio"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("visual", "cross", "text"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("visual", "cross", "speakers"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("text", "cross", "audio"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("text", "cross", "visual"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("text", "cross", "speakers"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("speakers", "cross", "audio"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("speakers", "cross", "visual"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("speakers", "cross", "text"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                    },
                    aggr="mean",
                )
            self.convs_1.append(convs)
            if not concat_head:
                self.bn = nn.BatchNorm1d(hidden_channels)
            elif concat_head:
                self.bn = nn.BatchNorm1d(hidden_channels * transformer_heads)
            self.conv_trans = TransformerConv(
                hidden_channels,
                hidden_channels,
                heads=transformer_heads,
                concat=concat_head,
            )
            self.dropout = nn.Dropout(drop_p)
            self.use_pair_norm = use_pair_norm
            if self.use_pair_norm:
                self.norm = PairNorm(norm_mode, norm_scale)

    def forward(self, x_dict, edge_index_dict, edge_weight_dict=None):
        # Change to device
        x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
        x_dict = {
            key: x.float() if x.dtype != torch.float32 else x
            for key, x in x_dict.items()
        }
        edge_index_dict = {k: v.to(self.device) for k, v in edge_index_dict.items()}
        for convs in self.convs_1:
            if self.drop_p > 0.0:
                x_dict = {key: self.dropout(x) for key, x in x_dict.items()}
            if edge_weight_dict is not None:
                x_dict = convs(x_dict, edge_index_dict, edge_weight_dict)
            elif edge_weight_dict is None:
                x_dict = convs(x_dict, edge_index_dict)
            # pair_norm here
            if self.use_pair_norm:
                x_dict = {key: self.norm(x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        if self.use_speaker in ["add", "no"]:
            homo_x, homo_edge_index, offset_audio, offset_text, offset_visual = (
                hetero_to_homo(x_dict, edge_index_dict)
            )
        elif self.use_speaker == "separate":
            (
                homo_x,
                homo_edge_index,
                offset_audio,
                offset_text,
                offset_visual,
                offset_speaker,
            ) = hetero_to_homo_4mod(x_dict, edge_index_dict)

        homo_x = nn.functional.leaky_relu(
            self.bn(self.conv_trans(homo_x, homo_edge_index))
        )
        if self.use_speaker in ["add", "no"]:
            homo_x, homo_edge_index, offset_audio, offset_text, offset_visual = (
                hetero_to_homo(x_dict, edge_index_dict)
            )
            homo_x = nn.functional.leaky_relu(
                self.bn(self.conv_trans(homo_x, homo_edge_index))
            )
            x_dict = {
                "audio": homo_x[offset_audio:offset_text],
                "text": homo_x[offset_text:offset_visual],
                "visual": homo_x[offset_visual:],
            }
        elif self.use_speaker == "separate":
            (
                homo_x,
                homo_edge_index,
                offset_audio,
                offset_text,
                offset_visual,
                offset_speaker,
            ) = hetero_to_homo_4mod(x_dict, edge_index_dict)
            homo_x = nn.functional.leaky_relu(
                self.bn(self.conv_trans(homo_x, homo_edge_index))
            )
            x_dict = {
                "audio": homo_x[offset_audio:offset_text],
                "text": homo_x[offset_text:offset_visual],
                "visual": homo_x[offset_visual:offset_speaker],
                "speakers": homo_x[offset_speaker:],
            }
        return x_dict


class FAConvHeteroBlock(nn.Module):
    def __init__(
        self,
        hidden_channels,
        device,
        num_layers,
        transformer_heads,
        use_speaker,
        concat_head,
        drop_p,
        use_pair_norm,
        norm_mode,
        norm_scale,
    ):
        super(FAConvHeteroBlock, self).__init__()
        self.device = device
        self.use_speaker = use_speaker
        self.convs_1 = nn.ModuleList()
        self.drop_p = drop_p
        for _ in range(num_layers):
            if use_speaker in ["add", "no"]:
                convs = HeteroConv(
                    {
                        ("audio", "past", "audio"): FAConv(
                            hidden_channels, hidden_channels
                        ),
                        ("visual", "past", "visual"): FAConv(
                            hidden_channels, hidden_channels
                        ),
                        ("text", "past", "text"): FAConv(
                            hidden_channels, hidden_channels
                        ),
                        ("audio", "future", "audio"): FAConv(
                            hidden_channels, hidden_channels
                        ),
                        ("visual", "future", "visual"): FAConv(
                            hidden_channels, hidden_channels
                        ),
                        ("text", "future", "text"): FAConv(
                            hidden_channels, hidden_channels
                        ),
                        ("audio", "self", "audio"): FAConv(
                            hidden_channels, hidden_channels
                        ),
                        ("visual", "self", "visual"): FAConv(
                            hidden_channels, hidden_channels
                        ),
                        ("text", "self", "text"): FAConv(
                            hidden_channels, hidden_channels
                        ),
                        ("audio", "cross", "visual"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("audio", "cross", "text"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("visual", "cross", "audio"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("visual", "cross", "text"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("text", "cross", "audio"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("text", "cross", "visual"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                    },
                    aggr="mean",
                )
            elif use_speaker == "separate":
                convs = HeteroConv(
                    {
                        ("audio", "past", "audio"): FAConv(
                            hidden_channels, hidden_channels
                        ),
                        ("visual", "past", "visual"): FAConv(
                            hidden_channels, hidden_channels
                        ),
                        ("text", "past", "text"): FAConv(
                            hidden_channels, hidden_channels
                        ),
                        ("speakers", "past", "speakers"): FAConv(
                            hidden_channels, hidden_channels
                        ),
                        ("audio", "future", "audio"): FAConv(
                            hidden_channels, hidden_channels
                        ),
                        ("visual", "future", "visual"): FAConv(
                            hidden_channels, hidden_channels
                        ),
                        ("text", "future", "text"): FAConv(
                            hidden_channels, hidden_channels
                        ),
                        ("speakers", "future", "speakers"): FAConv(
                            hidden_channels, hidden_channels
                        ),
                        ("audio", "self", "audio"): FAConv(
                            hidden_channels, hidden_channels
                        ),
                        ("visual", "self", "visual"): FAConv(
                            hidden_channels, hidden_channels
                        ),
                        ("text", "self", "text"): FAConv(
                            hidden_channels, hidden_channels
                        ),
                        ("speakers", "self", "speakers"): FAConv(
                            hidden_channels, hidden_channels
                        ),
                        ("audio", "cross", "visual"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("audio", "cross", "text"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("audio", "cross", "speakers"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("visual", "cross", "audio"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("visual", "cross", "text"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("visual", "cross", "speakers"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("text", "cross", "audio"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("text", "cross", "visual"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("text", "cross", "speakers"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("speakers", "cross", "audio"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("speakers", "cross", "visual"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                        ("speakers", "cross", "text"): GraphConv(
                            hidden_channels, hidden_channels
                        ),
                    },
                    aggr="mean",
                )

            self.convs_1.append(convs)
            if not concat_head:
                self.bn = nn.BatchNorm1d(hidden_channels)
            elif concat_head:
                self.bn = nn.BatchNorm1d(hidden_channels * transformer_heads)
            self.conv_trans = TransformerConv(
                hidden_channels,
                hidden_channels,
                heads=transformer_heads,
                concat=concat_head,
            )
            self.dropout = nn.Dropout(drop_p)
            self.use_pair_norm = use_pair_norm
            if self.use_pair_norm:
                self.norm = PairNorm(norm_mode, norm_scale)

    def forward(self, x_dict, edge_index_dict):
        # Change to device
        x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
        x_dict = {
            key: x.float() if x.dtype != torch.float32 else x
            for key, x in x_dict.items()
        }
        edge_index_dict = {k: v.to(self.device) for k, v in edge_index_dict.items()}
        for convs in self.convs_1:
            if self.drop_p > 0.0:
                x_dict = {key: self.dropout(x) for key, x in x_dict.items()}
            x_dict = convs(x_dict, edge_index_dict)
            if self.use_pair_norm:
                x_dict = {key: self.norm(x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        if self.use_speaker in ["add", "no"]:
            homo_x, homo_edge_index, offset_audio, offset_text, offset_visual = (
                hetero_to_homo(x_dict, edge_index_dict)
            )
        elif self.use_speaker == "separate":
            (
                homo_x,
                homo_edge_index,
                offset_audio,
                offset_text,
                offset_visual,
                offset_speaker,
            ) = hetero_to_homo_4mod(x_dict, edge_index_dict)

        homo_x = nn.functional.leaky_relu(
            self.bn(self.conv_trans(homo_x, homo_edge_index))
        )
        if self.use_speaker in ["add", "no"]:
            x_dict = {
                "audio": homo_x[offset_audio:offset_text],
                "text": homo_x[offset_text:offset_visual],
                "visual": homo_x[offset_visual:],
            }
        elif self.use_speaker == "separate":
            x_dict = {
                "audio": homo_x[offset_audio:offset_text],
                "text": homo_x[offset_text:offset_visual],
                "visual": homo_x[offset_visual:offset_speaker],
                "speakers": homo_x[offset_speaker:],
            }
        return x_dict


class highConvHeteroBlock(nn.Module):
    def __init__(
        self,
        hidden_channels,
        device,
        num_layers,
        transformer_heads,
        use_speaker,
        concat_head,
    ):
        super(highConvHeteroBlock, self).__init__()
        self.device = device
        self.use_speaker = use_speaker
        self.convs_1 = nn.ModuleList()
        for _ in range(num_layers):
            if use_speaker in ["add", "no"]:
                convs = HeteroConv(
                    {
                        ("audio", "past", "audio"): highConv(
                            hidden_channels, hidden_channels
                        ),
                        ("visual", "past", "visual"): highConv(
                            hidden_channels, hidden_channels
                        ),
                        ("text", "past", "text"): highConv(
                            hidden_channels, hidden_channels
                        ),
                        ("audio", "future", "audio"): highConv(
                            hidden_channels, hidden_channels
                        ),
                        ("visual", "future", "visual"): highConv(
                            hidden_channels, hidden_channels
                        ),
                        ("text", "future", "text"): highConv(
                            hidden_channels, hidden_channels
                        ),
                        ("audio", "self", "audio"): highConv(
                            hidden_channels, hidden_channels
                        ),
                        ("visual", "self", "visual"): highConv(
                            hidden_channels, hidden_channels
                        ),
                        ("text", "self", "text"): highConv(
                            hidden_channels, hidden_channels
                        ),
                        ("audio", "cross", "visual"): highConv(
                            hidden_channels, hidden_channels
                        ),
                        ("audio", "cross", "text"): highConv(
                            hidden_channels, hidden_channels
                        ),
                        ("visual", "cross", "audio"): highConv(
                            hidden_channels, hidden_channels
                        ),
                        ("visual", "cross", "text"): highConv(
                            hidden_channels, hidden_channels
                        ),
                        ("text", "cross", "audio"): highConv(
                            hidden_channels, hidden_channels
                        ),
                        ("text", "cross", "visual"): highConv(
                            hidden_channels, hidden_channels
                        ),
                    },
                    aggr="mean",
                )
            elif use_speaker == "separate":
                convs = HeteroConv(
                    {
                        ("audio", "past", "audio"): highConv(
                            hidden_channels, hidden_channels
                        ),
                        ("visual", "past", "visual"): highConv(
                            hidden_channels, hidden_channels
                        ),
                        ("text", "past", "text"): highConv(
                            hidden_channels, hidden_channels
                        ),
                        ("speakers", "past", "speakers"): highConv(
                            hidden_channels, hidden_channels
                        ),
                        ("audio", "future", "audio"): highConv(
                            hidden_channels, hidden_channels
                        ),
                        ("visual", "future", "visual"): highConv(
                            hidden_channels, hidden_channels
                        ),
                        ("text", "future", "text"): highConv(
                            hidden_channels, hidden_channels
                        ),
                        ("speakers", "future", "speakers"): highConv(
                            hidden_channels, hidden_channels
                        ),
                        ("audio", "self", "audio"): highConv(
                            hidden_channels, hidden_channels
                        ),
                        ("visual", "self", "visual"): highConv(
                            hidden_channels, hidden_channels
                        ),
                        ("text", "self", "text"): highConv(
                            hidden_channels, hidden_channels
                        ),
                        ("speakers", "self", "speakers"): highConv(
                            hidden_channels, hidden_channels
                        ),
                        ("audio", "cross", "visual"): highConv(
                            hidden_channels, hidden_channels
                        ),
                        ("audio", "cross", "text"): highConv(
                            hidden_channels, hidden_channels
                        ),
                        ("audio", "cross", "speakers"): highConv(
                            hidden_channels, hidden_channels
                        ),
                        ("visual", "cross", "audio"): highConv(
                            hidden_channels, hidden_channels
                        ),
                        ("visual", "cross", "text"): highConv(
                            hidden_channels, hidden_channels
                        ),
                        ("visual", "cross", "speakers"): highConv(
                            hidden_channels, hidden_channels
                        ),
                        ("text", "cross", "audio"): highConv(
                            hidden_channels, hidden_channels
                        ),
                        ("text", "cross", "visual"): highConv(
                            hidden_channels, hidden_channels
                        ),
                        ("text", "cross", "speakers"): highConv(
                            hidden_channels, hidden_channels
                        ),
                        ("speakers", "cross", "audio"): highConv(
                            hidden_channels, hidden_channels
                        ),
                        ("speakers", "cross", "visual"): highConv(
                            hidden_channels, hidden_channels
                        ),
                        ("speakers", "cross", "text"): highConv(
                            hidden_channels, hidden_channels
                        ),
                    },
                    aggr="mean",
                )
            self.convs_1.append(convs)
            if concat_head:
                self.bn = nn.BatchNorm1d(hidden_channels * transformer_heads)
            else:
                self.bn = nn.BatchNorm1d(hidden_channels)
            self.conv_trans = TransformerConv(
                hidden_channels,
                hidden_channels,
                heads=transformer_heads,
                concat=concat_head,
            )

    def forward(self, x_dict, edge_index_dict):
        # Change to device
        x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
        edge_index_dict = {k: v.to(self.device) for k, v in edge_index_dict.items()}
        for convs in self.convs_1:
            x_dict = convs(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        if self.use_speaker in ["add", "no"]:
            homo_x, homo_edge_index, offset_audio, offset_text, offset_visual = (
                hetero_to_homo(x_dict, edge_index_dict)
            )
        elif self.use_speaker == "separate":
            (
                homo_x,
                homo_edge_index,
                offset_audio,
                offset_text,
                offset_visual,
                offset_speaker,
            ) = hetero_to_homo_4mod(x_dict, edge_index_dict)

        homo_x = nn.functional.leaky_relu(
            self.bn(self.conv_trans(homo_x, homo_edge_index))
        )
        if self.use_speaker in ["add", "no"]:
            homo_x, homo_edge_index, offset_audio, offset_text, offset_visual = (
                hetero_to_homo(x_dict, edge_index_dict)
            )
            homo_x = nn.functional.leaky_relu(
                self.bn(self.conv_trans(homo_x, homo_edge_index))
            )
            x_dict = {
                "audio": homo_x[offset_audio:offset_text],
                "text": homo_x[offset_text:offset_visual],
                "visual": homo_x[offset_visual:],
            }
        elif self.use_speaker == "separate":
            (
                homo_x,
                homo_edge_index,
                offset_audio,
                offset_text,
                offset_visual,
                offset_speaker,
            ) = hetero_to_homo_4mod(x_dict, edge_index_dict)
            homo_x = nn.functional.leaky_relu(
                self.bn(self.conv_trans(homo_x, homo_edge_index))
            )
            x_dict = {
                "audio": homo_x[offset_audio:offset_text],
                "text": homo_x[offset_text:offset_visual],
                "visual": homo_x[offset_visual:offset_speaker],
                "speakers": homo_x[offset_speaker:],
            }
        return x_dict
