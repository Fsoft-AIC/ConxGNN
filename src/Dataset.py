import math
import random
import torch


class Dataset:

    def __init__(self, samples, args) -> None:
        self.args = args
        self.samples = samples
        self.batch_size = args.batch_size
        self.num_batches = math.ceil(len(self.samples) / args.batch_size)
        self.dataset = args.dataset
        self.speaker_to_idx = {"M": 0, "F": 1}
        self.embedding_dim = args.dataset_embedding_dims[args.dataset]
        self.int_classes, self.emo_classes, self.total_num_sentences = self.get_class_wise_num_sentences()
        self.metadata = {
            "batch_size": self.batch_size,
            "num_batches": self.num_batches,
            "dataset": self.dataset,
            "speaker_to_idx": self.speaker_to_idx,
            "embedding_dim": self.embedding_dim,
            "int_classes": self.int_classes,
            "emo_classes": self.emo_classes,
            "total_num_sentences": self.total_num_sentences
        }
        # Create the padded batches that is used for pair-wise cross-modality
        self.padded_batches = []
        for batch_idx in range(self.num_batches):
            batch = self.raw_batch(batch_idx)
            padded_batch = self.padding(batch)
            self.padded_batches.append(padded_batch)

    def get_class_wise_num_sentences(self):
        total_num_sentences = 0
        if self.args.dataset == "iemocap":
            int_classes = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            emo_classes = {
                "hap": 0,
                "sad": 0,
                "neu": 0,
                "ang": 0,
                "exc": 0,
                "fru": 0
            }
            dataset_label_dict = {
                0: "hap",
                1: "sad",
                2: "neu",
                3: "ang",
                4: "exc",
                5: "fru"
            }
        elif self.args.dataset == "iemocap_4":
            int_classes = {0: 0, 1: 0, 2: 0, 3: 0}
            emo_classes = {
                "hap": 0,
                "sad": 0,
                "neu": 0,
                "ang": 0
            }
            dataset_label_dict = {
                0: "hap",
                1: "sad",
                2: "neu",
                3: "ang"
            }
        elif self.args.dataset == "meld_m3net":
            # Mapping speakers to indices
            self.speaker_to_idx = {
                "0": 0,
                "1": 1,
                "2": 2,
                "3": 3,
                "4": 4,
                "5": 5,
                "6": 6,
                "7": 7,
                "8": 8
            }

            # Integer classes initialization
            int_classes = {
                0: 0,
                1: 0,
                2: 0,
                3: 0,
                4: 0,
                5: 0,
                6: 0
            }

            # Emotion classes initialization
            emo_classes = {
                "0": 0,
                "1": 0,
                "2": 0,
                "3": 0,
                "4": 0,
                "5": 0,
                "6": 0
            }

            # Dataset label dictionary
            dataset_label_dict = {
                0: "0",
                1: "1",
                2: "2",
                3: "3",
                4: "4",
                5: "5",
                6: "6",
            }
        valid_label = [key for key in dataset_label_dict.keys()]
        for dialog in self.samples:
            labels = dialog["labels"]  # this is a list that indicate the label
            for label in labels:
                assert label in valid_label, "label is invalid"
                int_classes[int(label)] += 1
                emo_classes[dataset_label_dict[int(label)]] += 1
                total_num_sentences += 1
        return int_classes, emo_classes, total_num_sentences

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        assert index < self.num_batches, "batch_idx %d > %d" % (
            index, self.num_batches)
        padded_batch = self.padded_batches[index]
        return padded_batch

    def raw_batch(self, index):
        assert index < self.num_batches, "batch_idx %d > %d" % (
            index, self.num_batches)
        batch = self.samples[index * self.batch_size:(index + 1) *
                             self.batch_size]
        return batch

    def padding(self, samples):
        # PreModel if data is meld_m3net
        batch_size = len(samples)
        # text_len_tensor is a tensor containing the lengths of the text sequences in each sample.
        text_len_tensor = torch.tensor([len(s["text"]) for s in samples]).long()
        # mx is the maximum length of these text sequences.
        mx = torch.max(text_len_tensor).item()

        text_tensor = torch.zeros((batch_size, mx, self.embedding_dim['t']))
        audio_tensor = torch.zeros((batch_size, mx, self.embedding_dim['a']))
        visual_tensor = torch.zeros((batch_size, mx, self.embedding_dim['v']))

        if self.args.dataset == "meld_m3net":
            roberta1_tensor = torch.zeros((batch_size, mx, self.embedding_dim['t1']))
            roberta2_tensor = torch.zeros((batch_size, mx, self.embedding_dim['t1']))
            roberta3_tensor = torch.zeros((batch_size, mx, self.embedding_dim['t1']))
            roberta4_tensor = torch.zeros((batch_size, mx, self.embedding_dim['t1']))

        speaker_tensor = torch.zeros((batch_size, mx)).long()
        labels = []
        utterances = []
        for i, s in enumerate(samples):
            cur_len = len(s["text"])
            utterances.append(s["sentence"])
            tmp_t = []
            tmp_a = []
            tmp_v = []
            for t, a, v in zip(s["text"], s["audio"], s["visual"]):
                tmp_t.append(torch.tensor(t))
                tmp_a.append(torch.tensor(a))
                tmp_v.append(torch.tensor(v))

            tmp_a = torch.stack(tmp_a)
            tmp_t = torch.stack(tmp_t)
            tmp_v = torch.stack(tmp_v)

            text_tensor[i, :cur_len, :] = tmp_t
            audio_tensor[i, :cur_len, :] = tmp_a
            visual_tensor[i, :cur_len, :] = tmp_v

            if self.args.dataset == "meld_m3net":

                tmp_t1 = []
                tmp_t2 = []
                tmp_t3 = []
                tmp_t4 = []
                for t1, t2, t3, t4 in zip(s["roberta1"], s["roberta2"], s["roberta3"], s["roberta4"]):
                    tmp_t1.append(torch.tensor(t1))
                    tmp_t2.append(torch.tensor(t2))
                    tmp_t3.append(torch.tensor(t3))
                    tmp_t4.append(torch.tensor(t3))
                tmp_t1 = torch.stack(tmp_t1)
                tmp_t2 = torch.stack(tmp_t2)
                tmp_t3 = torch.stack(tmp_t3)
                tmp_t4 = torch.stack(tmp_t4)

                roberta1_tensor[i, :cur_len, :] = tmp_t1
                roberta2_tensor[i, :cur_len, :] = tmp_t2
                roberta3_tensor[i, :cur_len, :] = tmp_t3
                roberta4_tensor[i, :cur_len, :] = tmp_t4
            if self.args.dataset in ["iemocap", "iemocap_4"]:
                speaker_tensor[i, :cur_len] = torch.tensor(
                    [self.speaker_to_idx[c] for c in s["speakers"]])
            elif self.args.dataset == "meld_m3net":
                speaker_tensor[i, :cur_len] = torch.tensor(
                    [self.speaker_to_idx[str(int(c))] for c in s["speakers"]])

            labels.extend(s["labels"])

        label_tensor = torch.tensor(labels).long()

        data = {
            "text_len_tensor": text_len_tensor,
            "text_tensor": text_tensor,
            "audio_tensor": audio_tensor,
            "visual_tensor": visual_tensor,
            "speakers_tensor": speaker_tensor,
            "label_tensor": label_tensor,
            "utterance_texts": utterances,
        }

        # Conditionally add attributes if the dataset is "meld_m3net"
        if self.args.dataset == "meld_m3net":
            data.update({
                "roberta1_tensor": roberta1_tensor,
                "roberta2_tensor": roberta2_tensor,
                "roberta3_tensor": roberta3_tensor,
                "roberta4_tensor": roberta4_tensor,
            })

        return data

    def shuffle(self):
        random.shuffle(self.samples)
