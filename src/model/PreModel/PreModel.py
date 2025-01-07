import torch.nn as nn


class PreModel(nn.Module):
    def __init__(self, feature_dim, norm_strategy="LN"):
        super(PreModel, self).__init__()
        self.feature_dim = feature_dim
        self.norm_strategy = norm_strategy

        # Define normalization layers
        self.normLNa = nn.LayerNorm(feature_dim)
        self.normLNb = nn.LayerNorm(feature_dim)
        self.normLNc = nn.LayerNorm(feature_dim)
        self.normLNd = nn.LayerNorm(feature_dim)

        self.normBNa = nn.BatchNorm1d(feature_dim)
        self.normBNb = nn.BatchNorm1d(feature_dim)
        self.normBNc = nn.BatchNorm1d(feature_dim)
        self.normBNd = nn.BatchNorm1d(feature_dim)

    def forward(self, data):
        # Assuming the dataset is meld_m3net and we have roberta tensors in the data
        r1 = data["roberta1_tensor"]
        r2 = data["roberta2_tensor"]
        r3 = data["roberta3_tensor"]
        r4 = data["roberta4_tensor"]

        B, max_len, feature_dim = r1.size()

        if self.norm_strategy == "LN":
            r1 = self.normLNa(r1.view(-1, feature_dim)).reshape(B, max_len, feature_dim)
            r2 = self.normLNb(r2.view(-1, feature_dim)).reshape(B, max_len, feature_dim)
            r3 = self.normLNc(r3.view(-1, feature_dim)).reshape(B, max_len, feature_dim)
            r4 = self.normLNd(r4.view(-1, feature_dim)).reshape(B, max_len, feature_dim)
        elif self.norm_strategy == "BN":
            r1 = self.normBNa(r1.view(-1, feature_dim)).reshape(B, max_len, feature_dim)
            r2 = self.normBNb(r2.view(-1, feature_dim)).reshape(B, max_len, feature_dim)
            r3 = self.normBNc(r3.view(-1, feature_dim)).reshape(B, max_len, feature_dim)
            r4 = self.normBNd(r4.view(-1, feature_dim)).reshape(B, max_len, feature_dim)
        elif self.norm_strategy == "LN2":
            norm2 = nn.LayerNorm((max_len, feature_dim), elementwise_affine=False)
            r1 = norm2(r1.view(-1, feature_dim)).reshape(B, max_len, feature_dim)
            r2 = norm2(r2.view(-1, feature_dim)).reshape(B, max_len, feature_dim)
            r3 = norm2(r3.view(-1, feature_dim)).reshape(B, max_len, feature_dim)
            r4 = norm2(r4.view(-1, feature_dim)).reshape(B, max_len, feature_dim)
        else:
            pass

        # Combine the tensors by averaging
        r = (r1 + r2 + r3 + r4) / 4
        data["text_tensor"] = r
        return data
