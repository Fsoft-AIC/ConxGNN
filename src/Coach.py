import copy
import time
import torch
from tqdm import tqdm
from sklearn import metrics
import src
import wandb

log = src.utils.get_logger()


class Coach:

    def __init__(self, trainset, devset, testset, model, opt, sched, args):
        self.trainset = trainset
        self.devset = devset
        self.testset = testset
        self.model = model
        self.opt = opt
        self.scheduler = sched
        self.args = args
        src.utils.check_uninitialized_parameters(model)
        src.utils.count_parameters(model)

        if self.args.wandb:
            assert self.args.prj_name is not None, "You must specify the argument --prj_name"
            assert self.args.run is not None, "You must specify the argument --run"
            wandb.require("core")
            self.wandb_logger = wandb.init(project=args.prj_name,
                                           name=args.run,
                                           config=args,
                                           reinit=True,
                                           settings=wandb.Settings(_disable_stats=True,
                                                                   _disable_meta=True))
            src.log_to_wandb(model)
        self.dataset_label_dict = {
            "iemocap": {
                "hap": 0,
                "sad": 1,
                "neu": 2,
                "ang": 3,
                "exc": 4,
                "fru": 5
            },
            "iemocap_4": {
                "hap": 0,
                "sad": 1,
                "neu": 2,
                "ang": 3
            },
            "mosei": {
                "Negative": 0,
                "Positive": 1
            },
            "meld_m3net": {
                "0": 0,
                "1": 1,
                "2": 2,
                "3": 3,
                "4": 4,
                "5": 5,
                "6": 6
            }
        }

        if args.emotion == "7class":
            self.label_to_idx = {
                "Strong Negative": 0,
                "Weak Negative": 1,
                "Negative": 2,
                "Neutral": 3,
                "Positive": 4,
                "Weak Positive": 5,
                "Strong Positive": 6,
            }
        else:
            self.label_to_idx = self.dataset_label_dict[args.dataset]

        self.best_dev_f1 = None
        self.best_epoch = None
        self.best_state = None

        # keep track of current epoch and iter
        self.args.training_status = {}
        self.args.training_status["cur_epoch"] = 1
        self.args.training_status["cur_iter"] = 1

    def load_ckpt(self, ckpt):
        self.best_dev_f1 = ckpt["best_dev_f1"]
        self.best_epoch = ckpt["best_epoch"]
        self.best_state = ckpt["best_state"]
        self.model.load_state_dict(self.best_state)
        print("Loaded model.....")

    def train(self):
        log.debug(self.model)
        # Early stopping.
        best_dev_f1, best_epoch, best_state = (
            self.best_dev_f1,
            self.best_epoch,
            self.best_state,
        )

        dev_f1s = []
        test_f1s = []
        train_losses = []
        best_test_f1 = None

        # Train
        for epoch in range(1, self.args.epochs + 1):
            self.args.training_status["cur_epoch"] = epoch
            if self.args.wandb:
                wandb.watch(
                    self.model,
                    log='all',
                    log_freq=int(self.args.trainset_metadata["num_batches"] *
                                 self.args.watch_log_every_n_epoch))
            train_loss, train_separate_losses = self.train_epoch(epoch)
            if self.args.training_status[
                    "cur_epoch"] > self.args.pretrain_unimodal_epochs:
                dev_f1, dev_loss, dev_separate_losses = self.evaluate()
                self.scheduler.step(dev_loss)
                test_f1, test_loss, test_separate_losses = self.evaluate(test=True)
                if self.args.wandb:
                    wandb.unwatch()
                log.info("[Dev set] [f1 {:.4f}]".format(dev_f1))
                if best_dev_f1 is None or dev_f1 > best_dev_f1:
                    best_dev_f1 = dev_f1
                    best_test_f1 = test_f1
                    best_epoch = epoch
                    best_state = copy.deepcopy(self.model.state_dict())
                    # if self.args.dataset == "mosei":
                    #     torch.save(
                    #         {
                    #             "args": self.args,
                    #             "state_dict": self.model
                    #         },
                    #         self.args.data_root +
                    #         "/model_checkpoints/mosei_best_dev_f1_model_" +
                    #         self.args.modalities + "_" + self.args.emotion + ".pt",
                    #     )
                    # else:
                    #     torch.save(
                    #         {
                    #             "args": self.args,
                    #             "state_dict": self.model
                    #         },
                    #         "." + "/model_checkpoints/" +
                    #         self.args.dataset + "_best_dev_f1_model_" +
                    #         self.args.modalities + ".pt",
                    #     )

                    log.info("Save the best model.")
                log.info("[Test set] [f1 {:.4f}]".format(test_f1))

                dev_f1s.append(dev_f1)
                test_f1s.append(test_f1)
                train_losses.append(train_loss)

                if self.args.wandb:
                    wandb.log({"train/loss": train_loss, "train/epoch": epoch})
                    wandb.log({"train/unimodal_loss": train_separate_losses["uni_modal_loss"]})
                    wandb.log({"train/CBContrastiveLoss": train_separate_losses["CBContrastiveLoss"]})
                    wandb.log({"train/CrossEntropyLoss": train_separate_losses["CrossEntropyLoss"]})

                    wandb.log({"dev/loss": dev_loss})
                    wandb.log({"dev/F1": dev_f1})
                    wandb.log({"dev/unimodal_loss": dev_separate_losses["uni_modal_loss"]})
                    wandb.log({"dev/CBContrastiveLoss": dev_separate_losses["CBContrastiveLoss"]})
                    wandb.log({"dev/CrossEntropyLoss": dev_separate_losses["CrossEntropyLoss"]})

                    wandb.log({"test/loss": test_loss})
                    wandb.log({"test/F1": test_f1})
                    wandb.log({"test/unimodal_loss": test_separate_losses["uni_modal_loss"]})
                    wandb.log({"test/CBContrastiveLoss": test_separate_losses["CBContrastiveLoss"]})
                    wandb.log({"test/CrossEntropyLoss": test_separate_losses["CrossEntropyLoss"]})

                    if self.args.use_hgr_loss:
                        wandb.log({"train/SoftHGRLoss": train_separate_losses["SoftHGRLoss"]})
                        wandb.log({"dev/SoftHGRLoss": dev_separate_losses["SoftHGRLoss"]})
                        wandb.log({"test/SoftHGRLoss": test_separate_losses["SoftHGRLoss"]})

        self.model.load_state_dict(best_state)
        log.info("")
        log.info("Best in epoch {}:".format(best_epoch))
        dev_f1, _, _ = self.evaluate()
        log.info("[Dev set] [f1 {:.4f}]".format(dev_f1))
        test_f1, _, _ = self.evaluate(test=True)
        log.info("[Test set] f1 {}".format(test_f1))
        if self.args.wandb:
            wandb.log({"best_dev_f1": best_dev_f1, "epoch": best_epoch})
            wandb.log({"best_test_f1": best_test_f1, "epoch": best_epoch})
        return best_dev_f1, best_test_f1, best_epoch, best_state, train_losses, dev_f1s, test_f1s

    def train_epoch(self, epoch):
        start_time = time.time()
        epoch_loss = 0
        epoch_separate_losses = {
            "uni_modal_loss": 0.0,
            "CBContrastiveLoss": 0.0,
            "CrossEntropyLoss": 0.0,
            "SoftHGRLoss": 0.0,
        }
        self.model.train()

        # self.trainset.shuffle()
        for idx in tqdm(range(len(self.trainset)), desc="train epoch {}".format(epoch)):
            self.model.zero_grad()
            data = self.trainset[idx]
            for k, v in data.items():
                if not k == "utterance_texts":
                    data[k] = v.to(self.args.device)
            nll, iter_separate_losses = self.model.get_loss(data)
            epoch_loss += nll.item()

            # updating separate losses
            epoch_separate_losses["uni_modal_loss"] += iter_separate_losses["uni_modal_loss"]
            if self.args.training_status["cur_epoch"] > self.args.pretrain_unimodal_epochs:
                epoch_separate_losses["CBContrastiveLoss"] += iter_separate_losses["CBContrastiveLoss"]
                epoch_separate_losses["CrossEntropyLoss"] += iter_separate_losses["CrossEntropyLoss"]
                if self.args.use_hgr_loss:
                    epoch_separate_losses["SoftHGRLoss"] += iter_separate_losses["SoftHGRLoss"]

            nll.backward()
            self.opt.step()

        end_time = time.time()
        log.info("")
        log.info("[Epoch %d] [Loss: %f] [Time: %f]" %
                 (epoch, epoch_loss, end_time - start_time))
        return epoch_loss, epoch_separate_losses

    def evaluate(self, test=False):
        eval_loss = 0
        epoch_separate_losses = {
            "uni_modal_loss": 0.0,
            "CBContrastiveLoss": 0.0,
            "CrossEntropyLoss": 0.0,
            "SoftHGRLoss": 0.0,
        }
        dataset = self.testset if test else self.devset
        self.model.eval()
        with torch.no_grad():
            golds = []
            preds = []
            for idx in tqdm(range(len(dataset)), desc="test" if test else "dev"):
                data = dataset[idx]
                golds.append(data["label_tensor"])
                for k, v in data.items():
                    if not k == "utterance_texts":
                        data[k] = v.to(self.args.device)
                y_hat = self.model(data)
                preds.append(y_hat.detach().to("cpu"))
                nll, iter_separate_losses = self.model.get_loss(data)
                eval_loss += nll.item()
                # updating separate losses
                epoch_separate_losses["uni_modal_loss"] += iter_separate_losses["uni_modal_loss"]
                if self.args.training_status["cur_epoch"] > self.args.pretrain_unimodal_epochs:
                    epoch_separate_losses["CBContrastiveLoss"] += iter_separate_losses["CBContrastiveLoss"]
                    epoch_separate_losses["CrossEntropyLoss"] += iter_separate_losses["CrossEntropyLoss"]
                    if self.args.use_hgr_loss:
                        epoch_separate_losses["SoftHGRLoss"] += iter_separate_losses["SoftHGRLoss"]

            golds = torch.cat(golds, dim=-1).cpu().numpy()
            preds = torch.cat(preds, dim=-1).cpu().numpy()
            f1 = metrics.f1_score(golds, preds, average="weighted")

            if test:
                print(
                    metrics.classification_report(
                        golds,
                        preds,
                        target_names=self.label_to_idx.keys(),
                        digits=4))

                if self.args.wandb:
                    # Log confusion matrix to wandb
                    wandb.log({
                        "confusion_matrix":
                        wandb.plot.confusion_matrix(
                            probs=None,
                            y_true=golds,
                            preds=preds,
                            class_names=list(self.label_to_idx.keys()))
                    })

        return f1, eval_loss, epoch_separate_losses
