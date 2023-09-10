import io
from typing import Any, Dict, Tuple

import numpy as np
import torch

from SRC_swig.LKH import OutputBetterTour, featureGenerate


def instance_string(instance: np.ndarray) -> str:
    with io.StringIO() as f:
        f.write("NAME : test \n")
        f.write("COMMENT : blank\n")
        f.write("TYPE : TSP\n")
        f.write("DIMENSION : " + str(len(instance)) + "\n")
        f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        s = 1000000
        for i in range(len(instance)):
            f.write(
                " "
                + str(i + 1)
                + " "
                + str(instance[i][0] * s)[:10]
                + " "
                + str(instance[i][1] * s)[:10]
                + "\n"
            )
        f.write("EOF\n")
        output_string = f.getvalue()
    return output_string


def candidate_pi_string(candidate: np.ndarray, pi: np.ndarray) -> Tuple[str, str]:
    n_node = candidate.shape[0]
    with io.StringIO() as f:
        f.write(str(n_node) + "\n")
        for j in range(n_node):
            line = str(j + 1) + " 0 5"
            for _ in range(5):
                line += " " + str(int(candidate[j, _])) + " " + str(_ * 100)
            f.write(line + "\n")
        f.write("-1\nEOF\n")
        candidate_string = f.getvalue()
    with io.StringIO() as f:
        f.write(str(n_node) + "\n")
        for j in range(n_node):
            line = str(j + 1) + " " + str(int(pi[j]))
            f.write(line + "\n")
        f.write("-1\nEOF\n")
        pi_string = f.getvalue()
    return candidate_string, pi_string


def param_string(
    time_limit: float = 10.0,
    seed: int = 1234,
    extra_para: Dict[str, Any] = None,
) -> str:
    with io.StringIO() as f:
        f.write("PROBLEM_FILE = NULL \n")
        f.write(f"TIME_LIMIT = {time_limit}\n")
        f.write("SEED = " + str(seed) + "\n")
        if extra_para is not None:
            for key, value in extra_para.items():
                f.write(key + " = " + str(value) + "\n")
        output_string = f.getvalue()
    return output_string


def primal_integral(gaps: np.ndarray, times: np.ndarray) -> float:
    # 计算 primal integral
    area = np.trapz(gaps, times)
    return area


def infer_SGN(
    net: torch.nn.Module,
    node_feat: np.ndarray,
    edge_index: np.ndarray,
    edge_feat: np.ndarray,
    inverse_edge_index: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    batch_size = node_feat.shape[0]
    device = next(net.parameters()).device
    node_feat = torch.FloatTensor(node_feat).to(device)
    edge_feat = torch.FloatTensor(edge_feat).to(device).view(batch_size, -1, 1)
    edge_index = torch.FloatTensor(edge_index).to(device).view(batch_size, -1)
    inverse_edge_index = (
        torch.FloatTensor(inverse_edge_index).to(device).view(batch_size, -1)
    )
    with torch.no_grad():
        y_edges, _, y_nodes = net.forward(
            node_feat, edge_feat, edge_index, inverse_edge_index, None, None, 20
        )
    y_edges = y_edges.detach().cpu().numpy()
    y_edges = y_edges[:, :, 1].reshape(batch_size, node_feat.shape[1], 20)
    y_edges = np.argsort(-y_edges, -1)
    edge_index = edge_index.cpu().numpy().reshape(-1, y_edges.shape[1], 20)
    candidate_index = edge_index[
        np.arange(batch_size).reshape(-1, 1, 1),
        np.arange(y_edges.shape[1]).reshape(1, -1, 1),
        y_edges,
    ]
    candidate = candidate_index[:, :, :5] + 1
    pi = y_nodes.cpu().numpy().squeeze(-1)
    return candidate.astype(int), pi


def generate_feat(
    data: np.ndarray, n_nodes: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    n_edges = 20
    invec = np.concatenate(
        [data.reshape(-1) * 1000000, np.zeros([n_nodes * (3 * n_edges - 2)])], -1
    )
    initial_tour = np.zeros([n_nodes], dtype=np.int32)
    feat_runtime = featureGenerate(1234, invec)
    OutputBetterTour(initial_tour)
    edge_index = invec[: n_nodes * n_edges].reshape(1, -1, 20)
    edge_feat = invec[n_nodes * n_edges : n_nodes * n_edges * 2].reshape(1, -1, 20)
    inverse_edge_index = invec[n_nodes * n_edges * 2 : n_nodes * n_edges * 3].reshape(
        1, -1, 20
    )
    initial_tour = initial_tour.reshape(1, -1)
    return (
        edge_index,
        edge_feat / 100000000,
        inverse_edge_index,
        feat_runtime / 1000000,
        initial_tour,
    )
