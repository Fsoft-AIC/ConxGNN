import torch.nn as nn
import copy
from .Classifier.Classifier import Classifier
from .Classifier.MultiEMOClassifier import ClassifierMultiEMO
from .UnimodalEncoder.UnimodalEncoder import UnimodalEncoder
from .Crossmodal.CrossmodalNet import CrossmodalNet
from .GraphModel.GraphModel import GraphModel
from ..utils import get_logger
from ..TensorGraph import TensorGraph
from .PreModel import PreModel

log = get_logger()


class MainModel(nn.Module):
    def __init__(self, args):
        super(MainModel, self).__init__()

        self.args = args
        self.modalities = args.modalities
        self.n_modals = len(self.modalities)
        self.use_speaker = args.use_speaker

        ic_dim = 0
        if self.args.use_speaker in ["no", "add"]:
            ic_dim = args.hidden_size * self.n_modals
        # if not args.use_graph_transformer and args.gcn_conv in ["gat_gcn", "gcn_gat"]:
        #     ic_dim = ic_dim * 2
        elif self.args.use_speaker == "separate":
            ic_dim = args.hidden_size * (self.n_modals + 1)
        if args.use_graph_transformer:
            ic_dim *= args.gnn_end_block_trans_heads
        if args.use_hyper_graph:
            ic_hyper_dim = 0
            if self.args.use_speaker in ["add", "no"]:
                ic_hyper_dim += args.hidden_size * self.n_modals
            elif self.args.use_speaker == "separate":
                ic_hyper_dim += args.hidden_size * (self.n_modals + 1)
            if args.use_hyper_attention and not args.use_custom_hyper:
                ic_hyper_dim *= args.hyper_end_trans_heads
            ic_dim += ic_hyper_dim
        if args.use_crossmodal and self.n_modals > 1:
            ic_dim += args.hidden_size * self.n_modals * (self.n_modals - 1)
        a_dim = args.dataset_embedding_dims[args.dataset]["a"]
        t_dim = args.dataset_embedding_dims[args.dataset]["t"]
        v_dim = args.dataset_embedding_dims[args.dataset]["v"]
        if args.dataset == "meld_m3net":
            t_dim = args.dataset_embedding_dims[args.dataset]["t1"]
        norm_strategy = args.norm_strategy
        dataset_label_dict = {
            "iemocap": {"hap": 0, "sad": 1, "neu": 2, "ang": 3, "exc": 4, "fru": 5},
            "iemocap_4": {"hap": 0, "sad": 1, "neu": 2, "ang": 3},
            "mosei": {"Negative": 0, "Positive": 1},
            "meld_m3net": {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6},
        }

        tag_size = len(dataset_label_dict[args.dataset])

        self.device = args.device
        if args.dataset == "meld_m3net":
            self.pre_model = PreModel(t_dim, norm_strategy)

        self.uniencoder = UnimodalEncoder(a_dim, t_dim, v_dim, args.hidden_size, args)

        ##########################################
        # start graph model and hypergraph model #
        ##########################################
        self.graph_model = GraphModel(args, ic_dim)

        if args.use_crossmodal and self.n_modals > 1:
            self.crossmodal = CrossmodalNet(args.hidden_size, args)
            print("CORECT --> Use Crossmodal")
        elif self.n_modals == 1:
            print("CORECT --> Crossmodal not available when number of modality is 1")
        if args.classifier_type == "corect":
            self.clf = Classifier(ic_dim, tag_size, args)
        elif args.classifier_type == "multiemo":
            self.clf = ClassifierMultiEMO(ic_dim, tag_size, args)
        self.rlog = {}

    def forward(self, data):
        batch_padded_tensor_data = copy.deepcopy(data)

        # PreModel if data is meld_m3net
        if self.args.dataset == "meld_m3net":
            batch_padded_tensor_data = self.pre_model(batch_padded_tensor_data)
        # Encoding multimodal feature
        base_tensor_graph = TensorGraph(batch_padded_tensor_data, self.modalities)
        tensor_graph = self.uniencoder(base_tensor_graph)
        # Graph construct : padded_dict2multimodal_features() automatically detects
        tensor_graph.padded_dict2multimodal_features()
        graph_out = self.graph_model(tensor_graph)
        if self.args.classifier_type == "corect":
            out = self.clf(graph_out, data["text_len_tensor"])
        elif self.args.classifier_type == "multiemo":
            out, classifier_custom_feats = self.clf(graph_out, data["text_len_tensor"])
        return out

    def get_loss(self, data):
        # Encoding multimodal feature
        total_loss = 0.0
        separate_losses = {}
        custom_feats = {}
        batch_padded_tensor_data = copy.deepcopy(data)
        # PreModel if data is meld_m3net
        if self.args.dataset == "meld_m3net":
            batch_padded_tensor_data = self.pre_model(batch_padded_tensor_data)
        base_tensor_graph = TensorGraph(batch_padded_tensor_data, self.modalities)
        uni_modal_loss, tensor_graph = self.uniencoder.get_loss(base_tensor_graph)
        if self.args.training_status["cur_epoch"] > self.args.pretrain_unimodal_epochs:
            # Graph construct : padded_dict2multimodal_features() automatically detects
            tensor_graph.padded_dict2multimodal_features()
            graph_out, _ = self.graph_model.get_loss(tensor_graph)
            if self.args.classifier_type == "corect":
                loss = self.clf.get_loss(
                    graph_out, data["label_tensor"], data["text_len_tensor"]
                )
            elif self.args.classifier_type == "multiemo":
                loss, classifier_separate_losses, classifier_custom_feats = (
                    self.clf.get_loss(
                        graph_out, data["label_tensor"], data["text_len_tensor"]
                    )
                )
                custom_feats["classifier_custom_feats"] = classifier_custom_feats
            total_loss = loss + self.args.InfoNCE_loss_param * uni_modal_loss
            separate_losses = classifier_separate_losses
        else:
            total_loss = uni_modal_loss

        separate_losses["uni_modal_loss"] = uni_modal_loss
        return total_loss, separate_losses

    def get_log(self):
        return self.rlog
