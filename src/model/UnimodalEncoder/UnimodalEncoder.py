import torch.nn as nn

from .EncoderModules import SeqTransfomer, LSTM_Layer
from .SpeakerEncoder import SpeakerEncoder, SpeakerEmbedding
from .utils import generate_dim_cutoff_embedding
from ...loss import compute_infonce_loss


class UnimodalEncoder(nn.Module):
    def __init__(self, a_dim, t_dim, v_dim, h_dim, args):
        super(UnimodalEncoder, self).__init__()
        self.hidden_dim = h_dim
        self.device = args.device
        self.rnn = args.rnn
        self.modalities = args.modalities
        self.a_dim = a_dim
        self.t_dim = t_dim
        self.v_dim = v_dim
        dataset_speaker_dict = {
            "iemocap": 2,
            "iemocap_4": 2,
            "mosei": 1,
            "meld_m3net": 9,
        }
        self.args = args
        self.use_speaker = args.use_speaker
        self.n_speakers = dataset_speaker_dict[args.dataset]

        if self.rnn != "m3net_like":
            self.audio_encoder = nn.Sequential(
                nn.Linear(self.a_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
            )
            self.vision_encoder = nn.Sequential(
                nn.Linear(self.v_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
            )
            if self.rnn == "transformer":
                self.text_encoder = SeqTransfomer(self.t_dim, self.hidden_dim, args)
                print("UnimodalEncoder --> Use transformer")
            elif self.rnn == "ffn":
                self.text_encoder = nn.Sequential(
                    nn.Linear(self.t_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.ReLU(),
                )
                print("UnimodalEncoder --> Use FNN")
            elif self.rnn == "lstm":
                self.text_encoder = LSTM_Layer(self.t_dim, self.hidden_dim, args)
                print("UnimodalEncoder --> Use LSTM")
        elif self.rnn == "m3net_like":
            self.audio_encoder = nn.Linear(self.a_dim, self.hidden_dim)
            self.vision_encoder = nn.Linear(self.v_dim, self.hidden_dim)
            self.text_encoder = nn.GRU(
                input_size=self.t_dim,
                hidden_size=self.hidden_dim // 2,
                num_layers=2,
                bidirectional=True,
            )
        if args.use_speaker == "add":
            self.speaker_embedding = SpeakerEmbedding(self.n_speakers, args.hidden_size)
        elif args.use_speaker == "separate":
            self.speaker_embedding = SpeakerEncoder(self.n_speakers, args.hidden_size)
        print(f"{args.dataset} speakers: {self.n_speakers}")

    def forward(self, ori_tensor_graph):
        # Encoding multimodal feature
        tensor_graph = ori_tensor_graph
        a = tensor_graph.data["audio_tensor"] if "a" in self.modalities else None
        t = tensor_graph.data["text_tensor"] if "t" in self.modalities else None
        v = tensor_graph.data["visual_tensor"] if "v" in self.modalities else None
        s = tensor_graph.data["speakers_tensor"]
        text_len_tensor = tensor_graph.metadata["text_len_tensor"]
        tensor_graph.data["audio_tensor"] = (
            self.audio_encoder(a) if a is not None else None
        )
        if t is None:
            tensor_graph.data["text_tensor"] = None
        else:
            if self.rnn == "lstm":
                tensor_graph.data["text_tensor"] = self.text_encoder(t, text_len_tensor)
            elif self.rnn == "m3net_like":
                tensor_graph.data["text_tensor"], hidden_text = self.text_encoder(t)
            else:
                tensor_graph.data["text_tensor"] = self.text_encoder(t)

        tensor_graph.data["visual_tensor"] = (
            self.vision_encoder(v) if v is not None else None
        )
        # Speaker embedding
        if self.use_speaker == "add":
            emb = self.speaker_embedding(tensor_graph.data["speakers_tensor"])
            if tensor_graph.data["audio_tensor"] is not None:
                tensor_graph.data["audio_tensor"] = (
                    tensor_graph.data["audio_tensor"] + emb
                )

            if tensor_graph.data["text_tensor"] is not None:
                tensor_graph.data["text_tensor"] = (
                    tensor_graph.data["text_tensor"] + emb
                )

            if tensor_graph.data["visual_tensor"] is not None:
                tensor_graph.data["visual_tensor"] = (
                    tensor_graph.data["visual_tensor"] + emb
                )

        elif self.use_speaker == "separate":
            s = self.speaker_embedding(s)
            tensor_graph.data["speakers_tensor"] = s

        return tensor_graph

    def get_loss(self, ori_tensor_graph):
        # Encoding multimodal feature
        tensor_graph = ori_tensor_graph
        a = tensor_graph.data["audio_tensor"] if "a" in self.modalities else None
        t = tensor_graph.data["text_tensor"] if "t" in self.modalities else None
        v = tensor_graph.data["visual_tensor"] if "v" in self.modalities else None
        s = tensor_graph.data["speakers_tensor"]
        text_len_tensor = tensor_graph.metadata["text_len_tensor"]

        # encoding original feats
        tensor_graph.data["audio_tensor"] = (
            self.audio_encoder(a) if a is not None else None
        )
        if t is None:
            tensor_graph.data["text_tensor"] = None
        else:
            if self.rnn == "lstm":
                tensor_graph.data["text_tensor"] = self.text_encoder(t, text_len_tensor)
            elif self.rnn == "m3net_like":
                tensor_graph.data["text_tensor"], hidden_text = self.text_encoder(t)
            else:
                tensor_graph.data["text_tensor"] = self.text_encoder(t)

        tensor_graph.data["visual_tensor"] = (
            self.vision_encoder(v) if v is not None else None
        )
        if self.args.use_contrastive_unimodal:
            # generate augmented feats :
            cutoff_a = (
                generate_dim_cutoff_embedding(
                    a, text_len_tensor, self.args.cutoff_ratio
                )
                if a is not None
                else None
            )
            cutoff_t = (
                generate_dim_cutoff_embedding(
                    t, text_len_tensor, self.args.cutoff_ratio
                )
                if t is not None
                else None
            )
            cutoff_v = (
                generate_dim_cutoff_embedding(
                    v, text_len_tensor, self.args.cutoff_ratio
                )
                if v is not None
                else None
            )

            cutoff_a_encoded = (
                self.audio_encoder(cutoff_a) if cutoff_a is not None else None
            )
            cutoff_t_encoded = (
                self.text_encoder(cutoff_t) if cutoff_t is not None else None
            )
            cutoff_v_encoded = (
                self.vision_encoder(cutoff_v) if cutoff_v is not None else None
            )
            # calculate infonce loss :
            contrast_a_loss = compute_infonce_loss(
                tensor_graph.data["audio_tensor"],
                cutoff_a_encoded,
                text_len_tensor,
                temperature=self.args.InfoNCE_temperature,
            )
            contrast_t_loss = compute_infonce_loss(
                tensor_graph.data["text_tensor"],
                cutoff_t_encoded,
                text_len_tensor,
                temperature=self.args.InfoNCE_temperature,
            )
            contrast_v_loss = compute_infonce_loss(
                tensor_graph.data["visual_tensor"],
                cutoff_v_encoded,
                text_len_tensor,
                temperature=self.args.InfoNCE_temperature,
            )

        # Speaker embedding
        if self.use_speaker == "add":
            emb = self.speaker_embedding(tensor_graph.data["speakers_tensor"])
            if tensor_graph.data["audio_tensor"] is not None:
                tensor_graph.data["audio_tensor"] = (
                    tensor_graph.data["audio_tensor"] + emb
                )

            if tensor_graph.data["text_tensor"] is not None:
                tensor_graph.data["text_tensor"] = (
                    tensor_graph.data["text_tensor"] + emb
                )

            if tensor_graph.data["visual_tensor"] is not None:
                tensor_graph.data["visual_tensor"] = (
                    tensor_graph.data["visual_tensor"] + emb
                )

        elif self.use_speaker == "separate":
            s = self.speaker_embedding(s)
            tensor_graph.data["speakers_tensor"] = s
        if self.args.use_contrastive_unimodal:
            loss = contrast_a_loss + contrast_v_loss
        else:
            loss = 0

        return loss, tensor_graph
