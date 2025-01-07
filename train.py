import argparse
import sys
import torch
import os
import src
import shutil
from utils import copy_repo_files

log = src.utils.get_logger()

current_dir = os.path.dirname(os.path.abspath(__file__))


def update_config(config, args):
    print(args)
    for key, value in vars(args).items():
        if value is not None:
            print(f"{key}\t{value}")
            keys = key.split(".")
            d = config
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value
    return config


def parse_args(config: argparse.Namespace):
    parser = argparse.ArgumentParser()
    for key, value in config.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, list):
                    parser.add_argument(
                        f"--{key}.{sub_key}", nargs="+", type=type(sub_value[0])
                    )
                else:
                    _type = int if isinstance(sub_value, bool) else type(sub_value)
                    parser.add_argument(f"--{key}.{sub_key}", type=_type)
        else:
            if isinstance(value, list):
                parser.add_argument(f"--{key}", nargs="+", type=type(value[0]))
            else:
                _type = int if isinstance(value, bool) else type(value)
                parser.add_argument(f"--{key}", type=_type)
    return parser


def main(args):
    src.utils.set_seed(args.seed)

    args.data = os.path.join(args.data_dir, "data_" + args.dataset + ".pkl")
    args.data_root = os.path.abspath(os.path.dirname(__file__))

    # load data
    log.debug("Loading data from '%s'." % args.data)

    data = src.utils.load_pkl(args.data)
    print(args.data)
    log.info("Loaded data.")

    trainset = src.Dataset(data["train"], args)
    devset = src.Dataset(data["dev"], args)
    testset = src.Dataset(data["test"], args)

    args["trainset_metadata"] = trainset.metadata
    args["devset_metadata"] = devset.metadata
    args["testset_metadata"] = testset.metadata

    log.debug("Building model...")

    model_file = "./model_checkpoints/model.pt"
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    model = src.MainModel(args).to(args.device)
    opt = src.Optim(args.learning_rate, args.max_grad_value, args.weight_decay)
    opt.set_parameters(model.parameters(), args.optimizer)
    sched = opt.get_scheduler(args.scheduler)

    coach = src.Coach(trainset, devset, testset, model, opt, sched, args)
    if not args.from_begin:
        ckpt = torch.load(model_file)
        coach.load_ckpt(ckpt)
        print("Training from checkpoint...")

    # Train
    log.info("Start training...")
    ret = coach.train()

    # Save.
    checkpoint = {
        "best_dev_f1": ret[0],
        "best_epoch": ret[1],
        "best_state": ret[2],
    }
    torch.save(checkpoint, model_file)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python train.py <config_file> [--param1 value1 --param2 value2 ...]"
        )
        sys.exit(1)

    config_file = sys.argv[1]
    config = src.utils.load_yaml(config_file)

    # Parse the remaining arguments after the config file
    if len(sys.argv) > 2:
        parser = parse_args(config)
        args = parser.parse_args(sys.argv[2:])
        config = update_config(config, args)
    print(config)
    config.__setitem__(
        "dataset_embedding_dims",
        {
            "iemocap": {
                "a": 100,
                "t": 768,
                "v": 512,
            },
            "iemocap_4": {
                "a": 100,
                "t": 768,
                "v": 512,
            },
            "mosei": {
                "a": 80,
                "t": 768,
                "v": 35,
            },
            "meld_m3net": {
                "a": 300,
                "t": 600,
                "v": 342,
                "t1": 1024,
                "t2": 1024,
                "t3": 1024,
                "t4": 1024,
            },
        },
    )
    print(config)
    log.debug(config)
    # Copy repo files before proceeding
    if config.backup is not None:
        dest_dir = os.path.join("backup_dir", config.backup)
        if os.path.exists(dest_dir):
            user_input = input(
                f"The directory '{dest_dir}' already exists. Do you want to override it?"
            ).strip()
            if user_input.lower() == "y":
                shutil.rmtree(dest_dir)  # Delete the existing directory
                print(f"Deleted the existing directory '{dest_dir}'.")
            else:
                dest_dir = user_input
        copy_repo_files(dest_dir=dest_dir, current_dir=current_dir)
        shutil.copy2(config_file, dest_dir)
    main(config)
