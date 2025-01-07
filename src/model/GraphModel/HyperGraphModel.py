import torch
import torch.nn as nn
from .GNNLayer.HypergraphConv import (
    HypergraphConv as hyper1,
)  # M3net custom hyper graph
from .GNNLayer.HypergraphConv2 import (
    HypergraphConv2 as hyper2,
)  # actual hypergraph from torch-geometric
from .utils import create_hyper_index


class HyperGraphModel(nn.Module):
    def __init__(self, args):
        super(HyperGraphModel, self).__init__()
        hidden_channels = args.hidden_size
        self.use_speaker = args.use_speaker
        self.device = args.device
        self.hyper_num_layers = args.hyper_num_layers
        self.use_custom_hyper = args.use_custom_hyper
        self.use_hyper_attention = args.use_hyper_attention
        for ll in range(self.hyper_num_layers):
            if self.use_custom_hyper:
                setattr(
                    self,
                    "hyperconv%d" % (ll + 1),
                    hyper1(hidden_channels, hidden_channels),
                )
            elif not self.use_custom_hyper and self.use_hyper_attention:
                if ll < self.hyper_num_layers - 1:
                    setattr(
                        self,
                        "hyperconv%d" % (ll + 1),
                        hyper2(
                            hidden_channels,
                            hidden_channels,
                            use_attention=True,
                            heads=args.hyper_trans_heads,
                            concat=False,
                        ),
                    )
                elif ll == self.hyper_num_layers - 1:
                    setattr(
                        self,
                        "hyperconv%d" % (ll + 1),
                        hyper2(
                            hidden_channels,
                            hidden_channels,
                            use_attention=True,
                            heads=args.hyper_end_trans_heads,
                            concat=True,
                        ),
                    )

        self.act_fn = nn.ReLU()
        self.hyperedge_weight = nn.Parameter(torch.ones(1000))
        self.EW_weight = nn.Parameter(torch.ones(5200))
        self.hyperedge_attr1 = nn.Parameter(torch.rand(hidden_channels))
        self.hyperedge_attr2 = nn.Parameter(torch.rand(hidden_channels))

    def forward(
        self, ori_tensor_graph
    ):  # must be in the form of node_concat_3mod or node_concat_4mod
        tensor_graph = ori_tensor_graph
        text_len_tensor = tensor_graph.metadata["text_len_tensor"]
        node_concat_feat = (
            tensor_graph.data
        )  # no need to deep copy cause this is tensor
        hyperedge_index, hyperedge_type1 = create_hyper_index(
            text_len_tensor, self.use_speaker, self.device
        )
        weight = self.hyperedge_weight[: hyperedge_index[1].max().item() + 1]
        EW_weight = self.EW_weight[: hyperedge_index.size(1)]
        edge_attr = self.hyperedge_attr1 * hyperedge_type1 + self.hyperedge_attr2 * (
            1 - hyperedge_type1
        )
        for ll in range(self.hyper_num_layers):
            if self.use_custom_hyper:
                node_concat_feat = getattr(self, "hyperconv%d" % (ll + 1))(
                    node_concat_feat, hyperedge_index, weight, edge_attr, EW_weight
                )
            else:
                node_concat_feat = getattr(self, "hyperconv%d" % (ll + 1))(
                    node_concat_feat, hyperedge_index, weight, edge_attr
                )
        tensor_graph.data = node_concat_feat
        return tensor_graph
