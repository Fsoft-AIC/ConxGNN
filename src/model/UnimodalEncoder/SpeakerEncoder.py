import torch
import math
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, dim):
        super(PositionalEncoding, self).__init__()
        self.dim = dim

    def forward(self, x):
        batch_size, num_node, _ = x.size()
        pe = torch.zeros(batch_size, num_node, self.dim, device=x.device)
        position = torch.arange(0,
                                num_node,
                                dtype=torch.float,
                                device=x.device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.dim, 2, device=x.device).float() *
            (-math.log(10000.0) / self.dim))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe


class SpeakerEmbedding(nn.Module):

    def __init__(self, n_speakers, dim):
        super(SpeakerEmbedding, self).__init__()
        self.embedding = nn.Embedding(
            n_speakers, dim)  # Binary speaker tensor, hence 2 classes

    def forward(self, x):
        batch_size, num_node = x.size()
        return self.embedding(x).view(batch_size, num_node, -1)


class SpeakerEncoder(nn.Module):

    def __init__(self, n_speakers, dim):
        super(SpeakerEncoder, self).__init__()
        self.pos_encoder = PositionalEncoding(dim)
        self.speaker_embedding = SpeakerEmbedding(n_speakers, dim)

    def forward(self, speaker_tensor):
        speaker_embedded = self.speaker_embedding(speaker_tensor)
        pos_encoded = self.pos_encoder(speaker_embedded)
        return pos_encoded + speaker_embedded
