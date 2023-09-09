import os
import sys

# Get the absolute path of the current script
current_script_path = os.path.abspath(__file__)

# Get the directory containing the current script
neurolkh_directory = os.path.dirname(current_script_path)
# Determine the parent directory (NeuroLKH folder) by going up one level
# neurolkh_directory = os.path.abspath(os.path.join(current_script_directory, ".."))

# Add the parent directory to sys.path
sys.path.append(neurolkh_directory)
sys.path.append(os.path.join(neurolkh_directory, "net"))

import os
import tempfile
from contextlib import contextmanager
from multiprocessing import Pool
from subprocess import check_call
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import tqdm
from torch.autograd import Variable

from lkh_utils import *
from sgcn_model import SparseGCNModel


@contextmanager
def change_dir(new_dir):
    # save old working directory
    old_dir = os.getcwd()
    try:
        # switch to new working directory
        os.chdir(new_dir)
        yield
    finally:
        # change back to previous working directory
        os.chdir(old_dir)


class LKHTransition:
    def __init__(
        self,
        model_path: str = "/home/xhpan/Codes/NeuroLKH/pretrained/neurolkh.pt",
        gpu_id: int = 0,
    ):
        self.net = SparseGCNModel()
        self.device = torch.device(
            f"cuda:{gpu_id}" if torch.cuda.is_available() and gpu_id >= 0 else "cpu"
        )
        self.net.to(self.device)
        saved = torch.load(model_path, map_location=self.device)
        self.net.load_state_dict(saved["model"])
        self.net.eval()
        del saved

    def get_candidate_pi_tour(self, instances: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Args:
            instances: [batch_size, num_nodes, 2]
        Returns:
            candidate: [batch_size, num_nodes, 5]
            pi: [batch_size, num_nodes]
            solutions: [batch_size, num_nodes]
        """
        assert instances.ndim == 3
        batch_size = instances.shape[0]
        with tempfile.TemporaryDirectory() as tmpdirname, change_dir(tmpdirname) as _:
            dataset_name = "tmp"
            mkdirs(dataset_name)
            processes = min(os.cpu_count(), batch_size)
            with Pool(processes) as pool:
                feats = (
                    pool.map(
                        generate_feat,
                        [
                            (dataset_name, instances[i], str(i))
                            for i in range(batch_size)
                        ],
                    )
                )
            feats = list(zip(*feats))
            edge_index, edge_feat, inverse_edge_index, _, tour = feats
            edge_index = np.concatenate(edge_index)
            edge_feat = np.concatenate(edge_feat)
            inverse_edge_index = np.concatenate(inverse_edge_index)
            tour = np.concatenate(tour)

        return (
            *infer_SGN(self.net, instances, edge_index, edge_feat, inverse_edge_index),
            tour,
        )

    def transition_function(
        self,
        state: Tuple[np.ndarray, ...],
        action: List[Dict[str, str]],
        time_limit: float = 1.0,
        max_trials: int = 10,
    ) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
        """
        Note that the node index here starts from 1.
        Args:
            state: (instances, solutions, candidates, pi)
                - instances: [batch_size, num_nodes, 2]
                - solutions: [batch_size, num_nodes]
                - candidates: [batch_size, num_nodes, k]
                - pi: [batch_size, num_nodes]
            action: a list of dict of hyperparameters
            time_limit: time limit for each instance
            max_trials: max trials for each instance
        Returns:
            next_state: (instances, solutions, candidates, pi)
            metrics: [batch_size,]
        """
        instances, solutions, candidates, pi = state
        batch_size = instances.shape[0]
        n_nodes = instances.shape[1]
        assert solutions.shape == (batch_size, n_nodes), solutions.shape
        # assert candidates.shape == (batch_size, n_nodes, 5), candidates.shape
        assert pi.shape == (batch_size, n_nodes), pi.shape
        assert len(action) == batch_size

        with tempfile.TemporaryDirectory() as tmpdirname, change_dir(tmpdirname) as _:
            # write instance
            dataset_name = "tmp"
            mkdirs(dataset_name)
            for i in range(batch_size):
                instance_name = str(i)
                instance_filename = (
                    "result/" + dataset_name + "/tsp/" + instance_name + ".tsp"
                )
                write_instance(instances[i], instance_name, instance_filename)
                write_candidate_pi(dataset_name, instance_name, candidates[i], pi[i])
                write_para(
                    dataset_name,
                    instance_name,
                    instance_filename,
                    "NeuroLKH",
                    "result/" + dataset_name + "/lkh_para/" + instance_name + ".para",
                    time_limit=time_limit,
                    max_trials=max_trials,
                    extra_para=action[i],
                )
                write_tour(
                    "result/" + dataset_name + "/init_tour/" + instance_name + ".tour",
                    solutions[i],
                )
            # run LKH
            processes = min(os.cpu_count(), batch_size)
            with Pool(processes) as pool:
                metrics = (
                    pool.map(
                        solve_NeuroLKH,
                        [
                            "result/" + dataset_name + "/lkh_para/" + str(i) + ".para"
                            for i in range(batch_size)
                        ],
                    )
                )
            # ! change metric scale
            metrics = np.array(metrics)
            new_candidates = []
            new_solutions = []
            with Pool(processes) as pool:
                for i in range(batch_size):
                    instance_name = str(i)
                    new_candidates.append(
                        pool.apply_async(
                            read_candidates,
                            (
                                "result/"
                                + dataset_name
                                + "/candidate/"
                                + instance_name
                                + ".txt",
                            ),
                        )
                    )
                    new_solutions.append(
                        pool.apply_async(
                            read_tour,
                            (
                                "result/"
                                + dataset_name
                                + "/tour/"
                                + instance_name
                                + ".tour",
                            ),
                        )
                    )
                new_candidates = [c.get() for c in new_candidates]
                new_solutions = [s.get().squeeze() for s in new_solutions]

            new_candidates = new_candidates
            new_solutions = np.stack(new_solutions)
        return (instances, new_solutions, new_candidates, pi), metrics
