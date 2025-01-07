import torch
import torch.nn as nn
from ...loss.SoftHGRLoss import SoftHGRLoss
from ...loss.FocalLoss import FocalLoss
from ...loss.CBConstrastiveLoss import CBContrastiveLoss
from .UnimodalSelfAttention import UnimodalSelfAttention
from .CrossModalAttention import CrossModalAttention
from .ModalWeighting import ModalWeighting


class ClassifierMultiEMO(nn.Module):
    def __init__(self, ic_dim, tag_size, args):
        super(ClassifierMultiEMO, self).__init__()
        self.use_sa = args.use_sa
        self.use_ca = args.use_ca
        hidden_size = args.hidden_size
        inter_size = args.inter_size
        use_class_weight = args.class_weight
        self.use_speaker = args.use_speaker
        sample_weight_param = args.sample_weight_param
        self.use_modal_weighting = args.use_modal_weighting
        self.modal_weighting_loss_param = args.modal_weighting_loss_param
        C_modal_weighting = args.C_modal_weighting
        classi_loss = args.classi_loss
        if args.use_speaker in ["add", "no"]:
            self.num_mod = 3
        elif args.use_speaker in ["separate"]:
            self.num_mod = 4
        self.UnimodalSelfAttention = UnimodalSelfAttention(
            embed_dim=inter_size, num_heads=args.sa_ca_heads, num_mod=self.num_mod
        )
        self.CrossModalAttention = CrossModalAttention(
            dim=inter_size, num_heads=args.sa_ca_heads, num_mod=self.num_mod
        )
        self.ModalWeighting = ModalWeighting(
            inter_size, tag_size, self.num_mod, C_modal_weighting
        )
        self.fc_layer = nn.Linear(int(inter_size * self.num_mod), hidden_size)
        self.mlp_layer = nn.Linear(hidden_size, tag_size)

        if not use_class_weight:
            if classi_loss == "CE":
                self.cls_loss = nn.CrossEntropyLoss()
            elif classi_loss == "Focal":
                self.cls_loss = FocalLoss()
        else:
            int_classes = args.trainset_metadata["int_classes"]
            """Assign more importance to minority classes."""
            class_weights = [
                (1 - sample_weight_param) / (1 - sample_weight_param**class_count)
                for class_count in torch.tensor(list(int_classes.values()))
            ]
            class_weights = torch.tensor(class_weights)
        if classi_loss == "CE":
            self.cls_loss = nn.CrossEntropyLoss(weight=class_weights) if use_class_weight else nn.CrossEntropyLoss()
        elif classi_loss == "Focal":
            self.cls_loss = FocalLoss()
        self.SoftHGRLoss = SoftHGRLoss()
        self.CBContrastiveLoss = CBContrastiveLoss(args)
        self.use_hgr_loss = args.use_hgr_loss
        self.HGR_loss_param = args.HGR_loss_param
        self.SWFC_loss_param = args.SWFC_loss_param
        self.CE_loss_param = args.CE_loss_param

    def forward(self, graph_out, text_len_tensor):
        # begin the self attention and cross attention
        # convert to multimodal_features : 3/4 padded B x max_len x dim
        graph_out.node_concat2multimodal_features()
        # self attention happen here
        if self.use_sa:
            graph_out, _ = self.UnimodalSelfAttention(graph_out)
        # cross attention happen here
        if self.use_ca:
            graph_out = self.CrossModalAttention(graph_out)

        graph_out.multimodal_features2node_concat()
        graph_out.node_concat2dim_concat()
        if self.use_modal_weighting:
            graph_out = self.ModalWeighting(graph_out)
        graph_out.data = self.fc_layer(graph_out.data)
        fc_outputs = torch.relu(graph_out.data)
        mlp_outputs = self.mlp_layer(fc_outputs)
        pred = torch.argmax(mlp_outputs, dim=-1)
        custom_feats = {"fc_outputs": fc_outputs, "mlp_outputs": mlp_outputs}
        return pred, custom_feats

    def get_loss(self, graph_out, label_tensor, text_len_tensor):  # now node_concat
        total_loss = 0.0
        separate_losses = {}
        custom_feats = {}
        if self.use_speaker in ["no", "add"]:
            n_node_per_modal = int(graph_out.data.size(0) // 3)
        elif self.use_speaker in ["separate"]:
            n_node_per_modal = int(graph_out.data.size(0) // 4)
        if self.use_hgr_loss:
            if self.use_speaker in ["no", "add"]:
                SoftHGRLoss = self.SoftHGRLoss(
                    graph_out.data[:n_node_per_modal, :],
                    graph_out.data[n_node_per_modal: n_node_per_modal * 2, :],
                    graph_out.data[n_node_per_modal * 2: n_node_per_modal * 3, :],
                )
            elif self.use_speaker == "separate":
                SoftHGRLoss = self.SoftHGRLoss(
                    graph_out.data[:n_node_per_modal, :],
                    graph_out.data[n_node_per_modal: n_node_per_modal * 2, :],
                    graph_out.data[n_node_per_modal * 2: n_node_per_modal * 3, :],
                    graph_out.data[n_node_per_modal * 3: n_node_per_modal * 4, :],
                )
        # begin the self attention and cross attention
        # convert to multimodal_features : 3/4 padded B x max_len x dim
        graph_out.node_concat2multimodal_features()
        # self attention happen here
        if self.use_sa:
            graph_out, _ = self.UnimodalSelfAttention(graph_out)
        # cross attention happen here
        if self.use_ca:
            graph_out = self.CrossModalAttention(graph_out)

        graph_out.multimodal_features2node_concat()
        graph_out.node_concat2dim_concat()
        if self.use_modal_weighting:
            weight_loss, graph_out = self.ModalWeighting.get_loss(
                graph_out, label_tensor
            )
        graph_out.data = self.fc_layer(graph_out.data)
        fc_outputs = torch.relu(graph_out.data)
        CBContrastiveLoss = self.CBContrastiveLoss(fc_outputs, label_tensor)
        mlp_outputs = self.mlp_layer(fc_outputs)
        cls_loss = self.cls_loss(mlp_outputs, label_tensor)
        total_loss = (
            CBContrastiveLoss * self.SWFC_loss_param + cls_loss * self.CE_loss_param
        )
        separate_losses = {
            "CBContrastiveLoss": CBContrastiveLoss,
            "CrossEntropyLoss": cls_loss,
        }
        if self.use_hgr_loss:
            total_loss = total_loss + SoftHGRLoss * self.HGR_loss_param
            separate_losses.update({"SoftHGRLoss": SoftHGRLoss})
        if self.use_modal_weighting:
            total_loss = total_loss + weight_loss * self.modal_weighting_loss_param
            separate_losses.update({"ModalWeightingLoss": weight_loss})
        custom_feats = {"fc_outputs": fc_outputs, "mlp_outputs": mlp_outputs}
        return total_loss, separate_losses, custom_feats
