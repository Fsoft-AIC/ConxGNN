import sys
import yaml
import random
import logging

import numpy as np
import torch
import pickle
import wandb
import torch.nn.parameter as param
from .Config import Config


def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return Config(data)


def set_seed(seed):
    """Sets random seed everywhere."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print("Seed set", seed)


def get_logger(level=logging.INFO):
    log = logging.getLogger(__name__)
    if log.handlers:
        return log
    log.setLevel(level)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S"
    )
    ch.setFormatter(formatter)
    log.addHandler(ch)
    return log


def save_pkl(obj, file):
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def load_pkl(file):
    with open(file, "rb") as f:
        return pickle.load(f)


def count_parameters(model, print_log=True):
    """Prints the number of total, trainable, and untrainable parameters in the model."""
    total_params = 0
    trainable_params = 0
    untrainable_params = 0

    for name, p in model.named_parameters():
        num_params = p.numel()
        total_params += num_params
        if p.requires_grad:
            trainable_params += num_params
        else:
            untrainable_params += num_params

    if print_log:
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        print(f"Untrainable parameters: {untrainable_params}")

    return total_params, trainable_params, untrainable_params


def log_to_wandb(model):

    total_params, learnable_params, unlearnable_params = count_parameters(model, print_log=False)

    wandb.log({
        "Total Parameters": total_params,
        "Learnable Parameters": learnable_params,
        "Unlearnable Parameters": unlearnable_params
    })

    # for name, p in model.named_parameters():
    #     wandb.log({
    #         f"Layer: {name}": {
    #             "Total": p.numel(),
    #             "Learnable": p.requires_grad
    #         }
    #     })


def check_uninitialized_parameters(model):
    for name, p in model.named_parameters():
        if isinstance(p, param.UninitializedParameter):
            print(f"Uninitialized parameter found: {name}")
