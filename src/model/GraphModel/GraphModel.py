import torch
import torch.nn as nn
from .HeteroGraphModel import HeteroGraphModel
from .HyperGraphModel import HyperGraphModel


class GraphModel(nn.Module):
    def __init__(self, args, ic_dim):
        super(GraphModel, self).__init__()
        self.args = args
        self.hetero = HeteroGraphModel(args)
        self.hyper = HyperGraphModel(args)
        if self.args.use_speaker in ["add", "no"]:
            self.num_mod = 3
        elif self.args.use_speaker in ["separate"]:
            self.num_mod = 4
        self.linear = nn.Linear(int(ic_dim / self.num_mod), args.inter_size)

    def forward(self, tensor_graph):
        out = []
        tensor_graph1 = tensor_graph.clone()
        # update the node embedding of batch_non_padded_hetero_data
        tensor_graph1.multimodal_features2BatchHeteroData()
        if self.args.use_hyper_graph:
            tensor_graph2 = tensor_graph1.clone()
            if self.num_mod == 3:
                tensor_graph2.BatchHeteroData2node_concat_3mod()
            elif self.num_mod == 4:
                tensor_graph2.BatchHeteroData2node_concat_4mod()
            hyper_graph = self.hyper(tensor_graph2)
        # tensor_graph1 will be used for graph model
        # tensor_graph2 will be used for hyper graph model
        hetero_graph = self.hetero(tensor_graph1)
        out.append(hetero_graph.data)  # node_concat representation
        if self.args.use_hyper_graph:
            out.append(hyper_graph.data)  # node_concat representation

        out = torch.cat(
            out, dim=-1
        )  # num_node * num_modality , (dim_hetero_graph + dim_hyper_graph)
        out = self.linear(out)  # num_node * num_modality , (intermediate_size)
        result_tensor_graph = tensor_graph.clone_metadata()
        result_tensor_graph.data = out
        if self.num_mod == 3:
            result_tensor_graph.representation = "node_concat_3mod"
        elif self.num_mod == 4:
            result_tensor_graph.representation = "node_concat_4mod"
        return result_tensor_graph

    def get_loss(self, tensor_graph):
        out = []
        tensor_graph1 = tensor_graph.clone()
        # update the node embedding of batch_non_padded_hetero_data
        tensor_graph1.multimodal_features2BatchHeteroData()
        if self.args.use_hyper_graph:
            tensor_graph2 = tensor_graph1.clone()
            if self.num_mod == 3:
                tensor_graph2.BatchHeteroData2node_concat_3mod()
            elif self.num_mod == 4:
                tensor_graph2.BatchHeteroData2node_concat_4mod()
            hyper_graph = self.hyper(tensor_graph2)
        # tensor_graph1 will be used for graph model
        # tensor_graph2 will be used for hyper graph model
        hetero_graph = self.hetero(tensor_graph1)
        out.append(hetero_graph.data)  # node_concat representation
        if self.args.use_hyper_graph:
            out.append(hyper_graph.data)  # node_concat representation
        out = torch.cat(
            out, dim=-1
        )  # num_node * num_modality , (dim_hetero_graph + dim_hyper_graph)
        out = self.linear(out)  # num_node * num_modality , (intermediate_size)
        result_tensor_graph = tensor_graph.clone_metadata()
        result_tensor_graph.data = out
        if self.num_mod == 3:
            result_tensor_graph.representation = "node_concat_3mod"
        elif self.num_mod == 4:
            result_tensor_graph.representation = "node_concat_4mod"
        loss = 0
        return result_tensor_graph, loss
