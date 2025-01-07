import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalWeighting(nn.Module):
    def __init__(self, dim, tag_size, num_mod, C_modal_weighting):
        super(ModalWeighting, self).__init__()
        self.c = C_modal_weighting  # Weight constant
        self.num_mod = num_mod
        self.tag_size = tag_size
        self.modality_weight = {
            "audio": 1.0,
            "text": 1.0,
            "visual": 1.0,
            "speakers": 1.0,
        }
        # Prediction network for each modality
        self.prediction_network = nn.ModuleDict(
            {
                "text": self._create_prediction_network(dim),
                "audio": self._create_prediction_network(dim),
                "visual": self._create_prediction_network(dim),
            }
        )
        # Weight mapping network for each modality
        self.weight_mapping_network = nn.ModuleDict(
            {
                "text": self._create_weight_mapping_network(dim),
                "audio": self._create_weight_mapping_network(dim),
                "visual": self._create_weight_mapping_network(dim),
            }
        )
        if num_mod == 4:
            self.prediction_network["speakers"] = self._create_prediction_network(
                dim, tag_size
            )
            self.weight_mapping_network["speakers"] = (
                self._create_weight_mapping_network(dim)
            )

    def _create_prediction_network(self, dim):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, self.tag_size),
        )

    def _create_weight_mapping_network(self, dim):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid(),  # Output is between 0 and 1
        )

    def calculate_l1_norm(self, predictions, ground_truth):
        # Convert ground truth to one-hot encoding
        ground_truth_one_hot = F.one_hot(
            ground_truth, num_classes=self.tag_size
        ).float()
        # Calculate the L1 norm (absolute difference) between predictions and ground truth one-hot
        l1_norm = torch.abs(predictions - ground_truth_one_hot).sum(dim=-1)
        return l1_norm

    def calculate_weights(self, errors):
        # Calculate the weights using the given formula
        exp_values = torch.exp(-errors * self.c)
        weights = exp_values / torch.sum(exp_values, dim=-1, keepdim=True)
        return weights

    def forward(self, tensor_graph):
        tensor = tensor_graph.data  # must be in dim_concat
        if self.num_mod == 3:
            dim = int(tensor.size(1) // 3)
            mod_list = ["text", "audio", "visual"]
        elif self.num_mod == 4:
            dim = int(tensor.size(1) // 4)
            mod_list = ["text", "audio", "visual", "speakers"]
        x = {
            "audio": tensor[:, :dim],
            "text": tensor[:, dim: dim * 2],
            "visual": tensor[:, dim * 2: dim * 3],
        }
        if self.num_mod == 4:
            x.update({"speakers": tensor[:, dim * 3: dim * 4]})
        weighted_tensor = []
        for modality in mod_list:
            x[modality] = x[modality] * self.modality_weight[modality]
            weighted_tensor.append(x[modality])
        tensor_graph.data = torch.cat(weighted_tensor, dim=-1)
        return tensor_graph

    def get_loss(self, tensor_graph, label_tensor):
        tensor = tensor_graph.data  # must be in dim_concat
        if self.num_mod == 3:
            dim = int(tensor.size(1) // 3)
            mod_list = ["text", "audio", "visual"]
        elif self.num_mod == 4:
            dim = int(tensor.size(1) // 4)
            mod_list = ["text", "audio", "visual", "speakers"]
        x = {
            "audio": tensor[:, :dim],
            "text": tensor[:, dim: dim * 2],
            "visual": tensor[:, dim * 2: dim * 3],
        }
        if self.num_mod == 4:
            x.update({"speakers": tensor[:, dim * 3: dim * 4]})
        errors = {"audio": 0.0, "text": 0.0, "visual": 0.0, "speakers": 0.0}
        weights = {"audio": 0.0, "text": 0.0, "visual": 0.0, "speakers": 0.0}
        for modality in mod_list:
            # get prediction for this modality:
            predictions = self.prediction_network[modality](x[modality])

            # Calculate L1 norm (errors) for each modality
            errors[modality] = self.calculate_l1_norm(predictions, label_tensor)
            # Calculate weights for each modality based on the errors
            weights[modality] = self.calculate_weights(errors[modality])

            # Predict the learned weights for each modality using the WeightMappingNetwork
            self.modality_weight[modality] = self.weight_mapping_network[modality](
                x[modality]
            )

        # Calculate the loss L_weight
        L_weight = sum(
            torch.abs(weights[modality] - self.modality_weight[modality]).mean()
            for modality in mod_list
        )
        weighted_tensor = []
        for modality in mod_list:
            x[modality] = x[modality] * self.modality_weight[modality]
            weighted_tensor.append(x[modality])
        tensor_graph.data = torch.cat(weighted_tensor, dim=-1)
        return L_weight, tensor_graph
