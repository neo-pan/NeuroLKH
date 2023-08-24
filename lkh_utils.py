from subprocess import check_call
import tempfile
import numpy as np
import os
import re

import torch


def read_candidates(filename):
    with open(filename, "r") as file:
        lines = file.readlines()

    num_nodes = int(lines[0])
    max_candidates = 0

    for line in lines[1 : 1 + num_nodes]:
        values = line.split()
        num_candidates = int(values[2])
        max_candidates = max(max_candidates, num_candidates)

    candidate_set = np.full((num_nodes, max_candidates), -1)

    for line in lines[1 : 1 + num_nodes]:
        values = line.split()
        node = int(values[0]) - 1
        num_candidates = int(values[2])

        for i in range(num_candidates):
            candidate_node = int(values[3 + 2 * i])
            candidate_set[node, i] = candidate_node

    return candidate_set


def read_tour(filename):
    tour = []

    with open(filename, "r") as file:
        lines = file.readlines()

    tour_section = False
    for line in lines:
        line = line.strip()

        if line == "TOUR_SECTION":
            tour_section = True
        elif line == "-1" or line == "EOF":
            tour_section = False
        elif tour_section:
            node = int(line)
            tour.append(node)
    tour = np.array(tour)

    return tour.reshape(1, -1)


def write_tour(filename, tour):
    with open(filename, "w") as file:
        file.write("TOUR_SECTION\n")
        for node in tour:
            file.write(str(node) + " ")
        file.write("-1\n")


def write_instance(instance, instance_name, instance_filename):
    with open(instance_filename, "w") as f:
        f.write("NAME : " + instance_name + "\n")
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


def write_candidate_pi(dataset_name, instance_name, candidate, pi):
    n_node = candidate.shape[0]
    with open(
        "result/" + dataset_name + "/candidate/" + instance_name + ".txt", "w"
    ) as f:
        f.write(str(n_node) + "\n")
        for j in range(n_node):
            line = str(j + 1) + " 0 5"
            for _ in range(5):
                line += " " + str(int(candidate[j, _])) + " " + str(_ * 100)
            f.write(line + "\n")
        f.write("-1\nEOF\n")
    with open("result/" + dataset_name + "/pi/" + instance_name + ".txt", "w") as f:
        f.write(str(n_node) + "\n")
        for j in range(n_node):
            line = str(j + 1) + " " + str(int(pi[j]))
            f.write(line + "\n")
        f.write("-1\nEOF\n")


def write_para(
    dataset_name,
    instance_name,
    instance_filename,
    method,
    para_filename,
    time_limit=1.0,
    max_trials=1000,
    seed=1234,
    extra_para=None,
):
    with open(para_filename, "w") as f:
        f.write("PROBLEM_FILE = " + instance_filename + "\n")
        f.write("MAX_TRIALS = " + str(max_trials) + "\n")
        # f.write("MOVE_TYPE = 5\nPATCHING_C = 3\nPATCHING_A = 2\nRUNS = 1\n")
        f.write(f"TIME_LIMIT = {time_limit}\n")
        f.write("SEED = " + str(seed) + "\n")
        f.write(
            "TOUR_FILE = " + "result/" + dataset_name + "/tour/" + instance_name + ".tour\n"
        )
        if method == "NeuroLKH":
            f.write("SUBGRADIENT = NO\n")
            f.write(
                "CANDIDATE_FILE = result/"
                + dataset_name
                + "/candidate/"
                + instance_name
                + ".txt\n"
            )
            f.write(
                "Pi_FILE = result/" + dataset_name + "/pi/" + instance_name + ".txt\n"
            )
            f.write(
                "INITIAL_TOUR_FILE = result/"
                + dataset_name
                + "/init_tour/"
                + instance_name
                + ".tour\n"
            )
        elif method == "FeatGenerate":
            f.write("GerenatingFeature\n")
            f.write(
                "Feat_FILE = result/"
                + dataset_name
                + "/feat/"
                + instance_name
                + ".txt\n"
            )
        else:
            raise NotImplementedError
        if extra_para is not None:
            for key, value in extra_para.items():
                f.write(key + " = " + str(value) + "\n")


def read_feat(feat_filename):
    edge_index = []
    edge_feat = []
    inverse_edge_index = []
    with open(feat_filename, "r") as f:
        lines = f.readlines()
        for line in lines[:-1]:
            line = line.strip().split()
            for i in range(20):
                edge_index.append(int(line[i * 3]))
                edge_feat.append(int(line[i * 3 + 1]) / 1000000)
                inverse_edge_index.append(int(line[i * 3 + 2]))
    edge_index = np.array(edge_index).reshape(1, -1, 20)
    edge_feat = np.array(edge_feat).reshape(1, -1, 20)
    inverse_edge_index = np.array(inverse_edge_index).reshape(1, -1, 20)
    runtime = float(lines[-1].strip())

    return edge_index, edge_feat, inverse_edge_index, runtime


def extract_gap_time(log_file):
    # Read the log file
    with open(log_file, "r") as f:
        log_text = f.read()

    # Extract the total number of runs
    match = re.search(r"RUNS = (\d+)", log_text)
    if match:
        total_runs = int(match.group(1))

    # Extract each trial and run's Gap and time
    trial_gaps = []
    trial_times = []
    run_data = {}
    current_run = 0
    prev_gap = None
    prev_time = None
    for match in re.finditer(
        r"(\* )?(\d+): Cost = \d+, Gap = ([\d.]+)%, Time = ([\d.]+) sec.", log_text
    ):
        is_trial = bool(match.group(1))
        trial = int(match.group(2))
        gap = float(match.group(3))
        time = float(match.group(4))

        if prev_gap is not None and prev_gap != gap:
            # Add additional points to make it a step function
            step_time = (time + prev_time) / 2
            trial_gaps.append(prev_gap)
            trial_times.append(time)

        prev_gap = gap
        prev_time = time

        if not is_trial:
            # Current run ends, store the data in the result dictionary
            trial_gaps.append(gap)
            trial_times.append(time)
            run_data[current_run] = {"gaps": trial_gaps, "times": trial_times}
            trial_gaps = []
            trial_times = []
            current_run += 1
            prev_gap = None
            prev_time = None
        else:
            trial_gaps.append(gap)
            trial_times.append(time)

    if current_run != total_runs:
        match = re.search(r"Successes/Runs = (\d+)/(\d+)", log_text)
        if match:
            successes = int(match.group(1))
            runs = int(match.group(2))
            if successes == 1 and runs == 0:
                return {0: {"gaps": [0], "times": [0]}}

        raise Warning(
            f"Number of runs {current_run} does not match total runs {total_runs}"
        )

    return run_data


def primal_integral(gaps, times):
    # 计算 primal integral
    area = np.trapz(gaps, times)
    return area


def primal_integral_mean(run_data):
    # 计算 primal integral 的平均值
    primal_integrals = []
    for run in run_data:
        primal_integrals.append(
            primal_integral(run_data[run]["gaps"], run_data[run]["times"])
        )
    primal_integrals = np.array(primal_integrals)
    return primal_integrals.mean()


def read_log(log_file):
    run_data = extract_gap_time(log_file)
    primal_integral_mean_value = primal_integral_mean(run_data)
    return primal_integral_mean_value


def mkdirs(dataset_name):
    os.makedirs("result/" + dataset_name + "/featgen_para", exist_ok=True)
    os.makedirs("result/" + dataset_name + "/feat", exist_ok=True)
    os.makedirs("result/" + dataset_name + "/tsp", exist_ok=True)
    os.makedirs("result/" + dataset_name + "/candidate", exist_ok=True)
    os.makedirs("result/" + dataset_name + "/pi", exist_ok=True)
    os.makedirs("result/" + dataset_name + "/init_tour", exist_ok=True)
    os.makedirs("result/" + dataset_name + "/tour", exist_ok=True)
    os.makedirs("result/" + dataset_name + "/lkh_para", exist_ok=True)
    os.makedirs("result/" + dataset_name + "/lkh_log", exist_ok=True)

def infer_SGN(net, node_feat, edge_index, edge_feat, inverse_edge_index):
    batch_size = node_feat.shape[0]
    device = next(net.parameters()).device
    node_feat = torch.FloatTensor(node_feat).to(device)
    edge_feat = torch.FloatTensor(edge_feat).to(device).view(batch_size, -1, 1)
    edge_index = torch.FloatTensor(edge_index).to(device).view(batch_size, -1)
    inverse_edge_index = torch.FloatTensor(inverse_edge_index).to(device).view(batch_size, -1)
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

def generate_feat(args):
    dataset_name, instance, instance_name = args
    para_filename = (
        "result/" + dataset_name + "/featgen_para/" + instance_name + ".para"
    )
    instance_filename = "result/" + dataset_name + "/tsp/" + instance_name + ".tsp"
    feat_filename = "result/" + dataset_name + "/feat/" + instance_name + ".txt"
    write_instance(instance, instance_name, instance_filename)
    write_para(
        dataset_name, instance_name, instance_filename, "FeatGenerate", para_filename
    )
    with tempfile.TemporaryFile() as f:
        check_call(["/home/xhpan/Codes/NeuroLKH/LKH", para_filename], stdout=f)

    init_tour = read_tour("result/" + dataset_name + "/tour/" + instance_name + ".tour")
    # init_tour = None

    return *read_feat(feat_filename), init_tour

def solve_NeuroLKH(para_filename):
    log_filename = para_filename.replace("lkh_para", "lkh_log").replace(".para", ".log")
    with open(log_filename, "w") as f:
        check_call(["/home/xhpan/Tools/LKH3/LKH-3.0.9/LKH", para_filename], stdout=f)
    return read_log(log_filename)
