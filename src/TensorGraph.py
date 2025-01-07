import torch
import copy
from torch_geometric.data import HeteroData, Batch
from torch import Tensor
from typing import Dict, List


class TensorGraph:
    def __init__(self, data: Dict = None, modalities: str = None):
        if (data is not None) and (modalities is not None):
            self.metadata = self.extract_metadata(data)
            self.set_data(data, self.metadata, modalities)
        else:
            self.metadata = {}
            self.data = None
            self.representation = None

    def clone(self):
        new_tensor_graph = TensorGraph()
        new_tensor_graph.metadata = self.metadata
        new_tensor_graph.representation = self.representation
        mutable_representations = [
            "padded_dict",
            "multimodal_features_3mod",
            "multimodal_features_4mod",
            "BatchHeteroData",
        ]
        immutable_representations = [
            "dim_concat_3mod",
            "dim_concat_4mod",
            "node_concat_3mod",
            "node_concat_4mod",
        ]
        if self.representation in mutable_representations:
            if self.representation == "BatchHeteroData":
                new_tensor_graph.data = self.data.clone()
            elif self.representation in [
                "multimodal_features_3mod",
                "multimodal_features_4mod",
            ]:
                new_tensor_graph.data = []
                for tensor in self.data:
                    new_tensor_graph.data.append(tensor.clone())
            else:
                new_tensor_graph.data = copy.deepcopy(self.data)
        elif self.representation in immutable_representations:
            new_tensor_graph.data = self.data
        return new_tensor_graph

    def clone_metadata(self):
        new_tensor_graph = TensorGraph()
        new_tensor_graph.metadata = self.metadata
        return new_tensor_graph

    @staticmethod
    def extract_metadata(batch_padded_data: Dict) -> Dict:
        metadata = {}
        metadata["text_len_tensor"] = batch_padded_data["text_len_tensor"]
        metadata["label_tensor"] = batch_padded_data["label_tensor"]
        metadata["utterance_texts"] = batch_padded_data["utterance_texts"]
        metadata["speakers_tensor"] = batch_padded_data["speakers_tensor"]
        return metadata

    def set_data(self, data, metadata, modalities):
        self.data = data
        batch_total_sentence = torch.sum(metadata["text_len_tensor"])
        if isinstance(data, torch.Tensor):
            assert (
                data.ndim == 2
            ), f"The # dimensions is {data.dim()}, but it should be 2"
            assert (
                data.shape[0] >= batch_total_sentence
            ), f"The shape of data is {data.shape[0]}, but it should >= {batch_total_sentence}"
            if data.shape[0] == batch_total_sentence:
                if len(modalities) == 3:
                    self.representation = "dim_concat_3mod"  # 'm*b/dim_a|dim_t|dim_v'
                elif len(modalities) == 4:
                    self.representation = (
                        "dim_concat_4mod"  # 'm*b/dim_a|dim_t|dim_v|dim_s'
                    )
            elif data.shape[0] > batch_total_sentence:
                if len(modalities) == 3:
                    self.representation = "node_concat_3mod"
                elif len(modalities) == 4:
                    self.representation = "node_concat_4mod"
        elif isinstance(data, Dict):
            self.data = data
            self.representation = "padded_dict"
        elif isinstance(data, List):
            self.data = data
            if len(modalities) == 3:
                self.representation = "multimodal_features_3mod"
            elif len(modalities) == 4:
                self.representation = "multimodal_features_4mod"
        elif isinstance(data, HeteroData) or isinstance(data, Batch):
            self.data = data
            self.representation = "BatchHeteroData"
        else:
            raise ValueError("Unsupported data type")

    def padded_dict2multimodal_features(self):
        assert (
            self.representation == "padded_dict"
        ), f"The representation is {self.representation}, but expected to be padded_dict"
        a = self.data["audio_tensor"]
        t = self.data["text_tensor"]
        v = self.data["visual_tensor"]
        s = self.data["speakers_tensor"]
        multimodal_features = []

        if a is not None:
            multimodal_features.append(a)
        if t is not None:
            multimodal_features.append(t)
        if v is not None:
            multimodal_features.append(v)
        if s.dim() == 3:
            multimodal_features.append(s)
            self.representation = "multimodal_features_4mod"
        elif s.dim() == 2:  # the speaker is still B x # max_num_node
            self.representation = "multimodal_features_3mod"
        self.data = multimodal_features

    def multimodal_features_3to4mod(self, speaker_tensor: Tensor) -> None:
        r"""
        The shape of speaker_tensor should be B x max_node x dim, meaning it must also be padded
        Usually the dim of speaker_tensor is just 1, but now you have to embed it using positional encoding and speaker embedding so that it have the same dim as other modalities
        """
        assert (
            self.representation == "multimodal_features_3mod"
        ), f"The representation is {self.representation}, but expected to be multimodal_features_3mod"
        self.representation = "multimodal_features_4mod"
        self.data.append(speaker_tensor)

    def multimodal_features_4to3mod(self) -> Tensor:
        assert (
            self.representation == "multimodal_features_4mod"
            f"The representation is {self.representation}, but expected to be multimodal_features_4mod"
        )
        speaker_tensor = self.data[3]
        self.data = [self.data[0], self.data[1], self.data[2]]
        self.representation == "multimodal_features_3mod"
        return speaker_tensor

    def multimodal_features2BatchHeteroData(self) -> None:
        r"""
        When we dont want to use the speaker node as a separate node, we use this instead of multimodal_features_4mod2BatchHeteroData
        """

        assert (
            self.representation
            in ["multimodal_features_3mod", "multimodal_features_4mod"]
        ), f"The representation is {self.representation}, but expected to be multimodal_features_3mod"
        batch_size = self.metadata["text_len_tensor"].size(0)
        hetero_data_list = []
        if self.representation == "multimodal_features_3mod":
            for j in range(batch_size):
                hetero_data = HeteroData()
                cur_len = self.metadata[
                    "text_len_tensor"
                ][j].item()
                hetero_data["audio"].x = self.data[0][j, :cur_len]
                hetero_data["text"].x = self.data[1][j, :cur_len]
                hetero_data["visual"].x = self.data[2][j, :cur_len]
                hetero_data["speakers"].x = self.metadata["speakers_tensor"][j, :cur_len]
                hetero_data_list.append(hetero_data)
        elif self.representation == "multimodal_features_4mod":
            for j in range(batch_size):
                hetero_data = HeteroData()
                cur_len = self.metadata["text_len_tensor"][j].item()
                hetero_data["audio"].x = self.data[0][j, :cur_len]
                hetero_data["text"].x = self.data[1][j, :cur_len]
                hetero_data["visual"].x = self.data[2][j, :cur_len]
                hetero_data["speakers"].x = self.data[3][j, :cur_len]
                hetero_data_list.append(hetero_data)
        batch_hetero_data = Batch.from_data_list(hetero_data_list)
        self.data = batch_hetero_data
        self.representation = "BatchHeteroData"

    def BatchHeteroData2dim_concat_3mod(self) -> None:
        assert (self.representation == "BatchHeteroData"), \
            f"The representation is {self.representation}, but expected to be BatchHeteroData"
        tensor = torch.cat(
            [self.data["audio"].x, self.data["text"].x, self.data["visual"].x], dim=1
        )
        self.representation = "dim_concat_3mod"
        self.data = tensor

    def BatchHeteroData2dim_concat_4mod(self) -> None:
        assert (self.representation == "BatchHeteroData"), \
            f"The representation is {self.representation}, but expected to be BatchHeteroData"
        tensor = torch.cat(
            [
                self.data["audio"].x,
                self.data["text"].x,
                self.data["visual"].x,
                self.data["speakers"].x,
            ],
            dim=1,
        )
        self.representation = "dim_concat_4mod"
        self.data = tensor

    def BatchHeteroData2node_concat_3mod(self) -> None:
        assert (self.representation == "BatchHeteroData"), \
            f"The representation is {self.representation}, but expected to be BatchHeteroData"
        tensor = torch.cat(
            [self.data["audio"].x, self.data["text"].x, self.data["visual"].x], dim=0
        )
        self.representation = "node_concat_3mod"
        self.data = tensor

    def BatchHeteroData2node_concat_4mod(self) -> None:
        assert (self.representation == "BatchHeteroData"), \
            f"The representation is {self.representation}, but expected to be BatchHeteroData"
        tensor = torch.cat(
            [
                self.data["audio"].x,
                self.data["text"].x,
                self.data["visual"].x,
                self.data["speakers"].x,
            ],
            dim=0,
        )
        self.representation = "node_concat_4mod"
        self.data = tensor

    def node_concat2dim_concat(self) -> None:
        assert (self.representation in ["node_concat_3mod", "node_concat_4mod"]), \
            f"The representation is {self.representation}, but expected to be node_concat_3mod or node_concat_4mod"
        if self.representation == "node_concat_3mod":
            num_utt = self.data.size(0) // 3
            audio_feat = self.data[0:num_utt]
            text_feat = self.data[num_utt: num_utt * 2]
            visual_feat = self.data[num_utt * 2:]
            tensor = torch.cat([audio_feat, text_feat, visual_feat], dim=1)
            self.representation = "dim_concat_3mod"
        elif self.representation == "node_concat_4mod":
            num_utt = self.data.size(0) // 4
            audio_feat = self.data[0:num_utt]
            text_feat = self.data[num_utt: num_utt * 2]
            visual_feat = self.data[num_utt * 2: num_utt * 3]
            speakers_feat = self.data[num_utt * 3:]
            tensor = torch.cat(
                [audio_feat, text_feat, visual_feat, speakers_feat], dim=1
            )
            self.representation = "dim_concat_4mod"
        self.data = tensor

    def node_concat2multimodal_features(self) -> None:
        assert (self.representation in ["node_concat_3mod", "node_concat_4mod"]), \
            f"The representation is {self.representation}, but expected to be node_concat_3mod or node_concat_4mod"

        # Get the device from the original data tensor
        device = self.data.device
        dim = self.data.size(1)

        if self.representation == "node_concat_3mod":
            num_utt = self.data.size(0) // 3
            batch_size = self.metadata["text_len_tensor"].size(0)

            # Splitting the features
            audio_feat = self.data[0:num_utt].to(device)
            text_feat = self.data[num_utt: num_utt * 2].to(device)
            visual_feat = self.data[num_utt * 2:].to(device)

            # Determine the max number of sentences in any dialogue
            max_n_sentence = torch.max(self.metadata["text_len_tensor"]).item()

            # Initialize zero tensors for each feature type on the same device
            audio_tensor = torch.zeros((batch_size, max_n_sentence, dim), device=device)
            text_tensor = torch.zeros((batch_size, max_n_sentence, dim), device=device)
            visual_tensor = torch.zeros(
                (batch_size, max_n_sentence, dim), device=device
            )

            # Current index tracker for features
            current_idx = 0

            for i in range(batch_size):
                n_sentences = self.metadata["text_len_tensor"][i].item()
                audio_tensor[i, :n_sentences] = audio_feat[current_idx: current_idx + n_sentences]
                text_tensor[i, :n_sentences] = text_feat[current_idx: current_idx + n_sentences]
                visual_tensor[i, :n_sentences] = visual_feat[current_idx: current_idx + n_sentences]
                current_idx += n_sentences
            self.data = [audio_tensor, text_tensor, visual_tensor]
            self.representation = "multimodal_features_3mod"

        elif self.representation == "node_concat_4mod":
            num_utt = self.data.size(0) // 4
            batch_size = self.metadata["text_len_tensor"].size(0)

            # Splitting the features
            audio_feat = self.data[0:num_utt].to(device)
            text_feat = self.data[num_utt: num_utt * 2].to(device)
            visual_feat = self.data[num_utt * 2: num_utt * 3].to(device)
            speakers_feat = self.data[num_utt * 3: num_utt * 4].to(device)

            # Determine the max number of sentences in any dialogue
            max_n_sentence = torch.max(self.metadata["text_len_tensor"]).item()

            # Initialize zero tensors for each feature type on the same device
            audio_tensor = torch.zeros((batch_size, max_n_sentence, dim), device=device)
            text_tensor = torch.zeros((batch_size, max_n_sentence, dim), device=device)
            visual_tensor = torch.zeros((batch_size, max_n_sentence, dim), device=device)
            speakers_tensor = torch.zeros((batch_size, max_n_sentence, dim), device=device)

            # Current index tracker for features
            current_idx = 0

            for i in range(batch_size):
                n_sentences = self.metadata["text_len_tensor"][i].item()
                audio_tensor[i, :n_sentences] = audio_feat[current_idx: current_idx + n_sentences]
                text_tensor[i, :n_sentences] = text_feat[current_idx: current_idx + n_sentences]
                visual_tensor[i, :n_sentences] = visual_feat[current_idx: current_idx + n_sentences]
                speakers_tensor[i, :n_sentences] = speakers_feat[current_idx: current_idx + n_sentences]
                current_idx += n_sentences
            self.data = [audio_tensor, text_tensor, visual_tensor, speakers_tensor]
            self.representation = "multimodal_features_4mod"

    def multimodal_features2node_concat(self) -> None:
        assert (self.representation in ["multimodal_features_3mod", "multimodal_features_4mod"]), \
            f"The representation is {self.representation}, but expected to be multimodal_features_3mod or multimodal_features_4mod"

        # Get the device from the original data tensor
        device = self.data[0].device
        dim = self.data[0].size(2)

        if self.representation == "multimodal_features_3mod":
            batch_size = self.data[0].size(0)
            max_n_sentence = self.data[0].size(1)
            num_utt = sum(self.metadata["text_len_tensor"])

            # Prepare the concatenated tensor
            node_concat = torch.zeros((num_utt * 3, dim), device=device)
            current_idx = 0

            for i in range(batch_size):
                n_sentences = self.metadata["text_len_tensor"][i].item()
                node_concat[current_idx: current_idx + n_sentences] = self.data[0][
                    i, :n_sentences, :
                ]
                node_concat[
                    current_idx + num_utt: current_idx + num_utt + n_sentences
                ] = self.data[1][i, :n_sentences, :]
                node_concat[
                    current_idx + 2 * num_utt: current_idx + 2 * num_utt + n_sentences
                ] = self.data[2][i, :n_sentences, :]
                current_idx += n_sentences

            self.data = node_concat
            self.representation = "node_concat_3mod"

        elif self.representation == "multimodal_features_4mod":
            batch_size = self.data[0].size(0)
            num_utt = sum(self.metadata["text_len_tensor"])

            # Prepare the concatenated tensor
            node_concat = torch.zeros((num_utt * 4, dim), device=device)
            current_idx = 0

            for i in range(batch_size):
                n_sentences = self.metadata["text_len_tensor"][i].item()
                node_concat[current_idx: current_idx + n_sentences] = self.data[0][
                    i, :n_sentences, :
                ]
                node_concat[
                    current_idx + num_utt: current_idx + num_utt + n_sentences
                ] = self.data[1][i, :n_sentences, :]
                node_concat[
                    current_idx + 2 * num_utt: current_idx + 2 * num_utt + n_sentences
                ] = self.data[2][i, :n_sentences, :]
                node_concat[
                    current_idx + 3 * num_utt: current_idx + 3 * num_utt + n_sentences
                ] = self.data[3][i, :n_sentences, :]
                current_idx += n_sentences

            self.data = node_concat
            self.representation = "node_concat_4mod"

    @staticmethod
    def static_node_concat2dim_concat(tensor, num_mod: int):
        num_utt = tensor.size(0) // num_mod
        audio_feat = tensor[0:num_utt]
        text_feat = tensor[num_utt: num_utt * 2]
        visual_feat = tensor[num_utt * 2: num_utt * 3]
        if num_mod == 3:
            result_tensor = torch.cat([audio_feat, text_feat, visual_feat], dim=1)
        elif num_mod == 4:
            speakers_feat = tensor[num_utt * 3:]
            result_tensor = torch.cat(
                [audio_feat, text_feat, visual_feat, speakers_feat], dim=1
            )
        return result_tensor
