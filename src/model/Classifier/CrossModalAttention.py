import torch.nn as nn


class CrossModalAttention(nn.Module):
    def __init__(self, dim, num_heads, num_mod):
        super(CrossModalAttention, self).__init__()
        self.multihead_attn_audio2text = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.multihead_attn_visual2text = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.multihead_attn_speakers2text = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.num_mod = num_mod

    def forward(self, tensor_graph):
        audio_feat = tensor_graph.data[0]
        text_feat = tensor_graph.data[1]
        visual_feat = tensor_graph.data[2]
        if self.num_mod == 4:
            speakers_feat = tensor_graph.data[3]
        # Audio -> Text Cross Attention
        audio_to_text_attention, _ = self.multihead_attn_audio2text(
            query=text_feat, key=audio_feat, value=audio_feat
        )

        # Visual -> Text Cross Attention
        visual_to_text_attention, _ = self.multihead_attn_visual2text(
            query=text_feat, key=visual_feat, value=visual_feat
        )

        # Speakers -> Text Cross Attention:
        if self.num_mod == 4:
            speakers_to_text_attention, _ = self.multihead_attn_speakers2text(
                query=text_feat, key=speakers_feat, value=speakers_feat
            )

        # Adding the attention to the original text features
        updated_text_feat = (
            text_feat + audio_to_text_attention + visual_to_text_attention
        )
        if self.num_mod == 4:
            updated_text_feat += speakers_to_text_attention
        tensor_graph.data = [audio_feat, updated_text_feat, visual_feat]
        return tensor_graph
