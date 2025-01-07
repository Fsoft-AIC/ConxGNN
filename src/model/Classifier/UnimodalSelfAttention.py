from torch import nn


class UnimodalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_mod):
        super(UnimodalSelfAttention, self).__init__()
        self.num_mod = num_mod
        # Initialize separate MultiheadAttention modules for each modality
        self.audio_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )
        self.text_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )
        self.visual_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )
        if self.num_mod == 4:
            # Optionally add speaker modality attention if required
            self.speaker_attention = nn.MultiheadAttention(
                embed_dim=embed_dim, num_heads=num_heads, batch_first=True
            )

    def forward(self, tensor_graph):
        # Assuming tensor_graph.data is a list of tensors [audio_tensor, text_tensor, visual_tensor]
        outputs = []
        weights = []

        # Apply attention separately to each tensor
        audio_output, audio_weights = self.audio_attention(
            tensor_graph.data[0], tensor_graph.data[0], tensor_graph.data[0]
        )
        outputs.append(audio_output)
        weights.append(audio_weights)

        text_output, text_weights = self.text_attention(
            tensor_graph.data[1], tensor_graph.data[1], tensor_graph.data[1]
        )
        outputs.append(text_output)
        weights.append(text_weights)

        visual_output, visual_weights = self.visual_attention(
            tensor_graph.data[2], tensor_graph.data[2], tensor_graph.data[2]
        )
        outputs.append(visual_output)
        weights.append(visual_weights)

        if self.num_mod == 4:
            speaker_output, speaker_weights = self.speaker_attention(
                tensor_graph.data[3], tensor_graph.data[3], tensor_graph.data[3]
            )
            outputs.append(speaker_output)
            weights.append(speaker_weights)

        tensor_graph.data = outputs
        return (
            tensor_graph,
            weights,
        )  # Optionally return attention weights for analysis or visualization
