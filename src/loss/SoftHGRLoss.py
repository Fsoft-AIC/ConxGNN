import torch
import torch.nn as nn
'''
Maximize the correlations across multimodal-fused features
extracted from MultiAttn through Soft-HGR loss.
'''


class SoftHGRLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def feature_mapping(self, feature_X, feature_Y):
        '''Calculate the inner products between feature mappings.'''
        feature_mapping_X_Y = torch.mean(torch.sum(feature_X * feature_Y, dim=-1), dim=0)

        return feature_mapping_X_Y

    def feature_covariance(self, feature_X, feature_Y):
        '''Calculate the inner products between feature covariances.'''
        cov_feature_X = torch.cov(feature_X)
        cov_feature_Y = torch.cov(feature_Y)
        # We empirically find that scaling the feature covariance by a factor of 1 / num_samples
        # leads to enhanced training stability and improvements in model performances.
        feature_covariance_X_Y = torch.trace(
            torch.matmul(cov_feature_X, cov_feature_Y))
        return feature_covariance_X_Y

    def forward(self, f_a, f_t, f_v, f_s=None):
        # num_samples is actually the B x num_nodes for one modality
        self.num_samples = f_t.shape[0]
        if f_s is None:
            all_features = [f_a, f_t, f_v]
        elif f_s is not None:
            all_features = [f_a, f_t, f_v, f_s]
        total_loss = 0.0
        for i in range(len(all_features) - 1):
            for j in range(i + 1, len(all_features)):
                feature_mapping_i_j = self.feature_mapping(all_features[i], all_features[j])
                feature_covariance_i_j = self.feature_covariance(all_features[i], all_features[j])
                soft_hgr_loss_i_j = feature_mapping_i_j - feature_covariance_i_j / 2
                total_loss += soft_hgr_loss_i_j

        loss = - total_loss / self.num_samples

        return loss
