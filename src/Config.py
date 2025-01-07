import copy


class Config:
    def __init__(self, dictionary):
        self._dict = {}
        for key, value in dictionary.items():
            self._dict[key] = value
        self.param_grouping = {}

    def __getitem__(self, item):
        return self._dict[item]

    def __getattr__(self, item):
        try:
            return self._dict[item]
        except KeyError:
            raise AttributeError(f"'Config' object has no attribute '{item}'")

    def __setitem__(self, key, value):
        self._dict[key] = value

    def setdefault(self, key, default):
        if key not in self._dict:
            self[key] = default
        return self._dict[key]

    def items(self):
        return self._dict.items()

    def __repr__(self):
        return str(self._dict)

    def copy(self):
        # Create a deep copy of the entire Config object
        new_config = Config(copy.deepcopy(self._dict))
        new_config.param_grouping = copy.deepcopy(self.param_grouping)
        return new_config

    def grouping_param(self):
        self.param_grouping["train"] = [
            # Common settings #
            ###################
            "dataset",
            "data_dir",
            "wandb",
            "prj_name",
            "run",
            "watch_log_every_n_epoch",
            "from_begin",
            "device",
            "epochs",
            "batch_size",
            "optimizer",
            "scheduler",
            "learning_rate",
            "emotion",
            "seed",
            "backup",
            "weight_decay",
            "max_grad_value",
            "drop_rate",
            # Model #
            #########
            # General #
            ###########
            "use_speaker",
            "modalities"
            # Hetero graph
            "wp_wf",
            "gnn_model",
            "gnn_block_trans_heads",
            "gnn_end_block_trans_heads",
            "edge_type",
            "use_graph_transformer",
            "gnn_block_num_layers",
            # Hyper graph
            "use_hyper_graph",
            "use_custom_hyper",
            "use_hyper_attention",
            "hyper_num_layers",
            "hyper_end_trans_heads",
            "hyper_trans_heads",
            # Unimodal encoder
            "hidden_size",
            "rnn",
            "encoder_nlayers",
            # Unimodal contrastive
            "use_contrastive_unimodal",
            "pretrain_unimodal_epochs",
            "InfoNCE_loss_param",
            "InfoNCE_temperature",
            "cutoff_ratio",
            "freeze_unimodal_after_pretraining",
            # Cross modal
            "use_crossmodal",
            "crossmodal_nheads",
            "num_crossmodal",
            "self_att_nheads",
            "num_self_att",
            # Classifier
            "class_weight",
            "classifier_type",
            "use_highway",
            "HGR_loss_param",
            "SWFC_loss_param",
            "CE_loss_param",
            "temp_param",
            "focus_param",
            "sample_weight_param",
        ]
        self.param_grouping["train_optuna"] = [
            # Common settings #
            ###################
            "dataset",
            "data_dir",
            "wandb",
            "prj_name",
            "run",
            "watch_log_every_n_epoch",
            "from_begin",
            "device",
            "epochs",
            "batch_size",
            "optimizer",
            "scheduler",
            "learning_rate",
            "emotion",
            "seed",
            "backup",
            "weight_decay",
            "max_grad_value",
            "drop_rate",
            # Model #
            #########
            # General #
            ###########
            "use_speaker",
            "modalities"
            # Hetero graph
            # these four are different :
            "wp",
            "wp_minus_wf",
            "gnn_num_block",
            "gnn_num_branch",
            "wp_wf",
            "gnn_model",
            "gnn_block_trans_heads",
            "gnn_end_block_trans_heads",
            "edge_type",
            "use_graph_transformer",
            "gnn_block_num_layers",
            # Hyper graph
            "use_hyper_graph",
            "use_custom_hyper",
            "use_hyper_attention",
            "hyper_num_layers",
            "hyper_end_trans_heads",
            "hyper_trans_heads",
            # Unimodal encoder
            "hidden_size",
            "rnn",
            "encoder_nlayers",
            # Unimodal contrastive
            "use_contrastive_unimodal",
            "pretrain_unimodal_epochs",
            "InfoNCE_loss_param",
            "InfoNCE_temperature",
            "cutoff_ratio",
            "freeze_unimodal_after_pretraining",
            # Cross modal
            "use_crossmodal",
            "crossmodal_nheads",
            "num_crossmodal",
            "self_att_nheads",
            "num_self_att",
            # Classifier
            "class_weight",
            "classifier_type",
            "use_highway",
            "HGR_loss_param",
            "SWFC_loss_param",
            "CE_loss_param",
            "temp_param",
            "focus_param",
            "sample_weight_param",
        ]
