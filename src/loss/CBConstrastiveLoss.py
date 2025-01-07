import torch
import torch.nn as nn
from torch.nn.functional import normalize
'''
Sample-Weighted Focal Contrastive (SWFC) Loss:
1. Divide training samples into positive and negative pairs to maximize
inter-class distances while minimizing intra-class distances;
2. Assign more importance to hard-to-classify positive pairs;
3. Assign more importance to minority classes.
'''


class CBContrastiveLoss(nn.Module):

    def __init__(self, args):
        '''
        temp_param: control the strength of penalty on hard negative samples;
        focus_param: forces the model to concentrate on hard-to-classify samples;
        sample_weight_param: control the strength of penalty on minority classes;
        dataset: MELD or IEMOCAP.
        device: cpu or cuda.
        '''
        super().__init__()
        self.temp_param = args.temp_param
        self.focus_param = args.focus_param
        self.sample_weight_param = args.sample_weight_param
        self.dataset = args.dataset
        self.total_counts = args.trainset_metadata["total_num_sentences"]
        self.int_classes = args.trainset_metadata["int_classes"]
        self.device = args.device

        if self.dataset == 'meld_m3net':
            self.num_classes = 9
            self.class_counts = torch.tensor(list(self.int_classes.values()))
        elif self.dataset == 'iemocap':
            self.num_classes = 6
            self.class_counts = torch.tensor(list(self.int_classes.values()))
            # self.class_counts = torch.tensor([458, 804, 1207, 913, 689, 1387])
        elif self.dataset == 'iemocap_4':
            self.num_classes = 4
            self.class_counts = torch.tensor(list(self.int_classes.values()))
        else:
            raise ValueError('Please choose either MELD or IEMOCAP')

        self.class_weights = self.get_sample_weights()

    def dot_product_similarity(self, current_features, feature_sets):
        '''Use dot-product to measure the similarity between feature pairs.'''
        similarity = torch.sum(current_features * feature_sets, dim=-1)
        similarity_probs = torch.softmax(similarity / self.temp_param, dim=0)

        return similarity_probs

    def positive_pairs_loss(self, similarity_probs):
        '''Calculate the loss contributed from positive pairs.'''
        pos_pairs_loss = torch.mean(torch.log(similarity_probs) *
                                    ((1 - similarity_probs)**self.focus_param),
                                    dim=0)

        return pos_pairs_loss

    def get_sample_weights(self):
        '''Assign more importance to minority classes.'''
        class_weights = [(1 - self.sample_weight_param) /
                         (1 - self.sample_weight_param**class_count)
                         for class_count in self.class_counts]
        return class_weights

    def forward(self, features, labels):
        self.num_samples = labels.shape[0]
        self.feature_dim = features.shape[-1]

        features = normalize(
            features,
            dim=-1)  # normalization helps smooth the learning process

        batch_sample_weights = torch.FloatTensor(
            [self.class_weights[label] for label in labels]).to(self.device)

        total_loss = 0.0
        for i in range(self.num_samples):
            current_feature = features[i]
            current_label = labels[i]
            feature_sets = torch.cat((features[:i], features[i + 1:]),
                                     dim=0)  # exclude the current feature
            label_sets = torch.cat((labels[:i], labels[i + 1:]),
                                   dim=0)  # exclude the current label
            expand_current_features = current_feature.expand(
                self.num_samples - 1, self.feature_dim).to(self.device)
            similarity_probs = self.dot_product_similarity(
                expand_current_features, feature_sets)
            pos_similarity_probs = similarity_probs[
                label_sets ==
                current_label]  # positive pairs with the same label
            if len(pos_similarity_probs) > 0:
                pos_pairs_loss = self.positive_pairs_loss(pos_similarity_probs)
                weighted_pos_pairs_loss = pos_pairs_loss * batch_sample_weights[
                    i]
                total_loss += weighted_pos_pairs_loss

        loss = -total_loss / self.num_samples

        return loss
