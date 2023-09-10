import functools
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
from multiprocessing import Pool
from typing import Dict, List, Tuple, Union

import numpy as np
import ParallelLKHSolver
import torch
from sgcn_model import SparseGCNModel
from torch.autograd import Variable

from lkh_utils import *


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
        ParallelLKHSolver.LKHSolver.debug = False
        self.lkh_solver = ParallelLKHSolver.LKHSolver()

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
        num_nodes = instances.shape[1]
        processes = min(os.cpu_count(), batch_size)
        with Pool(processes) as pool:
            feats = pool.map(
                functools.partial(generate_feat, n_nodes=num_nodes),
                [instances[i] for i in range(batch_size)],
            )
        feats = list(zip(*feats))
        edge_index, edge_feat, inverse_edge_index, _, tour = feats
        edge_index = np.concatenate(edge_index)
        edge_feat = np.concatenate(edge_feat)
        inverse_edge_index = np.concatenate(inverse_edge_index)
        tour = np.concatenate(tour)
        candidate, pi = infer_SGN(
            self.net, instances, edge_index, edge_feat, inverse_edge_index
        )
        # ParameterString, ProblemString, CandidateString, PenaltyString
        input_strings = [
            [
                param_string(),
                instance_string(instances[i]),
                *candidate_pi_string(candidate[i], pi[i]),
            ]
            for i in range(batch_size)
        ]
        self.lkh_solver.initialize(input_strings)

        return (
            candidate,
            pi,
            tour,
        )

    def transition_function(
        self,
        action: List[Dict[str, str]],
        trials: int = 10,
    ) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
        """
        Note that the node index here starts from 1.
        Args:
            action: a list of dict of hyperparameters
            max_trials: max trials for each instance
        Returns:
            next_state: (instances, solutions, candidates, pi)
            metrics: [batch_size,]
        """
        batch_size = len(action)
        assert (
            len(action) == self.lkh_solver.num_process
        ), f"{len(action)} != {self.lkh_solver.num_process}"

        gap_time_list = self.lkh_solver.trials(
            trials, [param_string(extra_para=a) for a in action]
        )
        metrics = np.zeros(batch_size, dtype=np.float64)
        for i in range(batch_size):
            metrics[i] = primal_integral(gap_time_list[i][1], gap_time_list[i][0])

        return metrics
