"""Loads configuration yaml and runs an experiment."""

from argparse import ArgumentParser
import os
from datetime import datetime
import shutil
import yaml
from tqdm import tqdm
import torch
import numpy as np

import data
import model
import probe
import regimen
import reporter
import task
import loss
from pathlib import Path


def choose_task_classes(args):
    """Chooses which task class to use based on config.

    Args:
      args: the global config dictionary built by yaml.
    Returns:
      A class to be instantiated as a task specification.
    """
    if args["probe"]["task_name"] == "parse-distance":
        task_class = task.ParseDistanceTask
        reporter_class = reporter.WordPairReporter
        if args["probe_training"]["loss"] == "L1":
            loss_class = loss.L1DistanceLoss
        else:
            raise ValueError(
                "Unknown loss type for given probe type: {}".format(
                    args["probe_training"]["loss"]
                )
            )
    elif args["probe"]["task_name"] == "parse-depth":
        task_class = task.ParseDepthTask
        reporter_class = reporter.WordReporter
        if args["probe_training"]["loss"] == "L1":
            loss_class = loss.L1DepthLoss
        else:
            raise ValueError(
                "Unknown loss type for given probe type: {}".format(
                    args["probe_training"]["loss"]
                )
            )
    else:
        raise ValueError(
            "Unknown probing task type: {}".format(args["probe"]["task_name"])
        )
    return task_class, reporter_class, loss_class


def choose_dataset_class(args):
    """Chooses which dataset class to use based on config.

    Args:
      args: the global config dictionary built by yaml.
    Returns:
      A class to be instantiated as a dataset.
    """
    if args["model"]["model_type"] in {
        "ELMo-disk",
        "ELMo-random-projection",
        "ELMo-decay",
    }:
        dataset_class = data.ELMoDataset
    elif args["model"]["model_type"] == "BERT-disk":
        dataset_class = data.BERTDataset
    elif args["model"]["model_type"] == "BERT-disk-cum":
        dataset_class = data.BERTCumulativeDataset
    elif args["model"]["model_type"] == "GPT2-disk-cum":
        dataset_class = data.GPT2CumulativeDataset
    elif args["model"]["model_type"] == "Llama3-disk-cum":
        dataset_class = data.Llama3CumulativeDataset
    else:
        raise ValueError(
            "Unknown model type for datasets: {}".format(args["model"]["model_type"])
        )

    return dataset_class


def choose_probe_class(args):
    """Chooses which probe and reporter classes to use based on config.

    Args:
      args: the global config dictionary built by yaml.
    Returns:
      A probe_class to be instantiated.
    """
    if args["probe"]["task_signature"] == "word":
        if args["probe"]["psd_parameters"]:
            return probe.OneWordPSDProbe
        else:
            return probe.OneWordNonPSDProbe
    elif args["probe"]["task_signature"] == "word_pair":
        if args["probe"]["psd_parameters"]:
            if args["probe"]["cumulative"]:
                return probe.TwoWordPSDCumulativeProbe
            else:
                return probe.TwoWordPSDProbe
        else:
            if args["probe"]["cumulative"]:
                raise ValueError(
                    "Cumulative probe not implemented for non-PSD word pairs"
                )
            else:
                return probe.TwoWordNonPSDProbe
    else:
        raise ValueError(
            "Unknown probe type (probe function signature): {}".format(
                args["probe"]["task_signature"]
            )
        )


def choose_model_class(args):
    """Chooses which reporesentation learner class to use based on config.

    Args:
      args: the global config dictionary built by yaml.
    Returns:
      A class to be instantiated as a model to supply word representations.
    """
    if args["model"]["model_type"] == "ELMo-disk":
        return model.DiskModel
    elif (
        args["model"]["model_type"] == "BERT-disk"
        or args["model"]["model_type"] == "BERT-disk-cum"
        or args["model"]["model_type"] == "GPT2-disk-cum"
        or args["model"]["model_type"] == "Llama3-disk-cum"
    ):
        return model.DiskModel
    elif args["model"]["model_type"] == "ELMo-random-projection":
        return model.ProjectionModel
    elif args["model"]["model_type"] == "ELMo-decay":
        return model.DecayModel
    elif args["model"]["model_type"] == "pytorch_model":
        raise ValueError("Using pytorch models for embeddings not yet supported...")
    else:
        raise ValueError("Unknown model type: {}".format(args["model"]["model_type"]))


def run_train_probe(args, probe, dataset, model, loss, reporter, regimen):
    """Trains a structural probe according to args.

    Args:
      args: the global config dictionary built by yaml.
            Describes experiment settings.
      probe: An instance of probe.Probe or subclass.
            Maps hidden states to linguistic quantities.
      dataset: An instance of data.SimpleDataset or subclass.
            Provides access to DataLoaders of corpora.
      model: An instance of model.Model
            Provides word representations.
      reporter: An instance of reporter.Reporter
            Implements evaluation and visualization scripts.
    Returns:
      None; causes probe parameters to be written to disk.
    """
    regimen.train_until_convergence(
        probe, model, loss, dataset.get_train_dataloader(), dataset.get_dev_dataloader()
    )


def run_report_results(
    args, probe, dataset, model, loss, reporter, regimen, test_only=False
):
    """
    Reports results from a structural probe according to args.
    By default, does so only for dev set.
    Requires a simple code change to run on the test set.
    """
    probe_params_path = os.path.join(
        args["reporting"]["root"], args["probe"]["params_path"]
    )
    probe.load_state_dict(torch.load(probe_params_path))
    probe.eval()

    if not test_only:
        dev_dataloader = dataset.get_dev_dataloader()
        dev_predictions = regimen.predict(probe, model, dev_dataloader)
        if hasattr(probe, "center_of_gravity"):
            state = {
                "center_of_gravity": probe.center_of_gravity(),
                "scaling_factor": probe.scaling.item(),
                "weights": probe.weights.tolist(),
            }
        else:
            state = {}
        reporter(dev_predictions, dev_dataloader, "dev", state)

    # Uncomment to run on the test set
    test_dataloader = dataset.get_test_dataloader()
    test_predictions = regimen.predict(probe, model, test_dataloader)
    if hasattr(probe, "center_of_gravity"):
        state = {
            "center_of_gravity": probe.center_of_gravity(),
            "scaling_factor": probe.scaling.item(),
            "weights": probe.weights.tolist(),
        }
    else:
        state = {}
    if "extra_test_output_name" in args:
        output_name = args["extra_test_output_name"]
    else:
        output_name = "test"
    reporter(test_predictions, test_dataloader, output_name, state)


def execute_experiment(args, train_probe, report_results):
    """
    Execute an experiment as determined by the configuration
    in args.

    Args:
      train_probe: Boolean whether to train the probe
      report_results: Boolean whether to report results
    """
    probe_params_path = os.path.join(
        args["reporting"]["root"], args["probe"]["params_path"]
    )
    if train_probe and os.path.exists(probe_params_path):
        print("Probe already trained, skipping experiment...")
        return
    dataset_class = choose_dataset_class(args)
    task_class, reporter_class, loss_class = choose_task_classes(args)
    probe_class = choose_probe_class(args)
    model_class = choose_model_class(args)
    regimen_class = regimen.ProbeRegimen

    task = task_class()
    expt_dataset = dataset_class(args, task)
    expt_reporter = reporter_class(args)
    expt_probe = probe_class(args)
    expt_model = model_class(args)
    expt_regimen = regimen_class(args)
    expt_loss = loss_class(args)

    if train_probe:
        print("Training probe...")
        run_train_probe(
            args,
            expt_probe,
            expt_dataset,
            expt_model,
            expt_loss,
            expt_reporter,
            expt_regimen,
        )
    if report_results:
        print("Reporting results of trained probe...")
        if "extra_test_output_name" in args:
            print("Reporting results of trained probe on extra test set...")
            args["reporting"]["observation_paths"][
                "test_path"
            ] = f"{args['extra_test_output_name']}.observations"
            args["reporting"]["prediction_paths"][
                "test_path"
            ] = f"{args['extra_test_output_name']}.predictions"
        test_only = True if "extra_test_output_name" in args else False
        run_report_results(
            args,
            expt_probe,
            expt_dataset,
            expt_model,
            expt_loss,
            expt_reporter,
            expt_regimen,
            test_only,
        )


# def setup_new_experiment_dir(args, yaml_args):
#   """Constructs a directory in which results and params will be stored.

#   If reuse_results_path is not None, then it is reused; no new
#   directory is constrcted.

#   Args:
#     args: the command-line arguments:
#     yaml_args: the global config dictionary loaded from yaml
#     reuse_results_path: the (optional) path to reuse from a previous run.
#   """
#   now = datetime.now()
#   date_suffix = '-'.join((str(x) for x in [now.year, now.month, now.day, now.hour, now.minute, now.second, now.microsecond]))
#   model_suffix = '-'.join((yaml_args['model']['model_type'], yaml_args['probe']['task_name']))
#   try:
#     shutil.copyfile(args.model_config, os.path.join(yaml_args['reporting']['root'], os.path.basename(args.model_config)))
#   except shutil.SameFileError:
#     tqdm.write('Note, the config being used is the same as that already present in the results dir')

if __name__ == "__main__":
    argp = ArgumentParser()
    argp.add_argument("model_config")
    argp.add_argument(
        "--train-probe", action="store_true", help="Set to train a new probe. "
    )
    argp.add_argument(
        "--report-results",
        default=1,
        type=int,
        help="Set to report results; " "(optionally after training a new probe)",
    )
    argp.add_argument(
        "--seed",
        default=0,
        type=int,
        help="sets all random seeds for (within-machine) reproducibility",
    )
    argp.add_argument("--corpus_root", default="data", help="")
    argp.add_argument("--corpus_train_path", required=True, help="")
    argp.add_argument("--corpus_dev_path", required=True, help="")
    argp.add_argument("--corpus_test_path", required=True, help="")
    argp.add_argument("--embeddings_root", default="data", help="")
    argp.add_argument("--embeddings_train_path", required=True, help="")
    argp.add_argument("--embeddings_dev_path", required=True, help="")
    argp.add_argument("--embeddings_test_path", required=True, help="")
    argp.add_argument("--output_dir", required=True, help="")
    argp.add_argument("--extra_test_output_name", required=False, default=None, help="")
    cli_args = argp.parse_args()
    if cli_args.seed:
        np.random.seed(cli_args.seed)
        torch.manual_seed(cli_args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    yaml_args = yaml.safe_load(open(cli_args.model_config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yaml_args["device"] = device
    yaml_args["dataset"]["corpus"] = {
        "root": os.path.abspath(os.path.abspath(cli_args.corpus_root)),
        "train_path": cli_args.corpus_train_path,
        "dev_path": cli_args.corpus_dev_path,
        "test_path": cli_args.corpus_test_path,
    }
    yaml_args["dataset"]["embeddings"]["root"] = os.path.abspath(
        os.path.abspath(cli_args.embeddings_root)
    )
    yaml_args["dataset"]["embeddings"]["train_path"] = cli_args.embeddings_train_path
    yaml_args["dataset"]["embeddings"]["dev_path"] = cli_args.embeddings_dev_path
    yaml_args["dataset"]["embeddings"]["test_path"] = cli_args.embeddings_test_path
    yaml_args["reporting"]["root"] = os.path.join(
        os.path.abspath(cli_args.output_dir),
        f"layer{yaml_args['model']['model_layer']}",
        f"seed{cli_args.seed}",
    )
    Path(yaml_args["reporting"]["root"]).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        cli_args.model_config,
        os.path.join(
            yaml_args["reporting"]["root"], os.path.basename(cli_args.model_config)
        ),
    )
    if cli_args.extra_test_output_name:
        yaml_args["extra_test_output_name"] = cli_args.extra_test_output_name
    print(yaml_args)
    execute_experiment(
        yaml_args,
        train_probe=cli_args.train_probe,
        report_results=cli_args.report_results,
    )
