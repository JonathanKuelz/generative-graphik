import itertools
from multiprocessing import Pool, cpu_count
from typing import Callable, Dict, Optional, Union

from liegroups.numpy.se3 import SE3Matrix
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader

from graphik.graphs import ProblemGraphRevolute
from graphik.robots import RobotRevolute
from graphik.utils import graph_from_pos
from generative_graphik.utils.dataset_generation import generate_data_point_from_pose, create_dataset_from_data_points
from generative_graphik.utils.get_model import get_model
from generative_graphik.utils.torch_to_graphik import joint_transforms_to_t_zero


def _default_cost_function(T_desired: SE3Matrix, T_eef: SE3Matrix) -> torch.Tensor:
    """
    The default cost function for the inverse kinematics problem. It is the sum of the squared errors between the
    desired and actual end-effector poses.

    :param T_desired: The desired end-effector pose.
    :param T_eef: The actual end-effector pose.
    :return: The cost.
    """
    err = T_desired.inv().dot(T_eef)
    cos_angle = 0.5 * np.trace(err.rot.mat) - 0.5
    cos_angle = np.clip(cos_angle, -1., 1.)  # avoid NaNs from rounding errors

    angle = np.arccos(cos_angle) * (180 / np.pi)  # degree
    trans = np.linalg.norm(err.trans)
    return trans + angle * 0.005  # 2° ~ 1cm


def _get_goal_idx(num_robots, samples_per_robot, batch_size, num_batch, idx_batch):
    num_sample = num_batch * batch_size + idx_batch
    return num_sample % samples_per_robot

def _get_robot_idx(num_robots, samples_per_robot, batch_size, num_batch, idx_batch):
    num_sample = num_batch * batch_size + idx_batch
    return num_sample // samples_per_robot


def get_q(p_problem, graph, goal, compute_cost, ik_cost_function=_default_cost_function):
    """
    Retrieve the joint angles for a single problem.

    :param p_problem: The position data for the problem of shape n_samples x num_nodes x dim.
    :param graph: The graph object for the problem.
    :param goal: The goal for the problem as SE3Matrix.
    :param compute_cost: If True, computes the cost for each sample.
    :param ik_cost_function: The cost function to use if compute_cost is True. Needs to accept two SE3Matrix objects.
    """
    q_all, cost_all = [], []
    eef = graph.robot.end_effectors[-1]
    joint_ids = graph.robot.joint_ids[1:]
    for sample, p in enumerate(p_problem):
        q = graph.joint_variables(graph_from_pos(p, graph.node_ids), {eef: goal})
        q_all.append([q[key] for key in joint_ids])
        if compute_cost:
            T_ee = graph.robot.pose(q, eef)
            cost_all.append(ik_cost_function(goal, T_ee))
        else:
            cost_all.append(0)
    return q_all, cost_all


def _get_q_batch(P_all: torch.Tensor,
                graphs: Dict[str, ProblemGraphRevolute],
                goals: np.ndarray,
                ik_cost_function: Callable,
                batch_size: int,
                return_all: bool,
                i: int,
                nR: int,
                nG: int,
                n_proc: Union[int, str] = 'auto'
                ) -> None:
    """
    Helper function to parallelize the retrieval of joint angles from position data
    :param P_all: The position data of batch size x num_samples x num_nodes x dim
    :param graphs: The graph objects for all problems
    :param goals: The goals for all problems as numpy array
    :param ik_cost_function: The cost function to use if only the best sample shall be returned
    :param batch_size: The original batch size, needed to obtain the correct graph and goal
    :param return_all: If True, returns all the samples from the forward pass, if false, returns only the best one.
    :param i: The index of the current batch
    :param nR: The number of robots
    :param nG: The number of goals
    :param n_proc: The maximum number of processes to use. If auto, uses half of the available cores.
    """
    device = P_all.device
    if n_proc == 'auto':
        n_proc = cpu_count() // 2

    args = [[pos,
             graphs[_get_robot_idx(nR, nG, batch_size, i, j)],
             SE3Matrix.from_matrix(goals[_get_robot_idx(nR, nG, batch_size, i, j), _get_goal_idx(nR, nG, batch_size, i, j)], normalize=True),
             not return_all, ik_cost_function] for j, pos in enumerate(P_all.detach().cpu().numpy())]

    n_proc = min(n_proc, len(args))
    if n_proc > 1:
        with Pool(n_proc) as p:
            res = p.starmap(get_q, args)
    else:
        res = [get_q(*arg) for arg in args]

    q_res, cost_res = map(lambda x: torch.tensor(x, device=device), zip(*res))
    if return_all:
        return q_res
    else:
        best_idx = torch.argmin(cost_res, dim=1)
        return q_res[torch.arange(q_res.shape[0]), best_idx, ...]


def ik(kinematic_chains: torch.tensor,
       goals: torch.tensor,
       samples: int = 16,
       return_all: bool = False,
       ik_cost_function: Callable = _default_cost_function,
       batch_size: int = 64,
       num_processes_get_q: int = 8
       ) -> torch.Tensor:
    """
    This function takes robot kinematics and any number of goals and solves the inverse kinematics, using graphIK.

    :param kinematic_chains: A tensor of shape (nR, N, 4, 4) containing the joint transformations of nR robots with N
        joints each.
    :param goals: A tensor of shape (nR, nG, 4, 4) containing the desired end-effector poses.
    :param samples: The number of samples to use for the forward pass of the model.
    :param return_all: If True, returns all the samples from the forward pass, so the resulting tensor has a shape
        nR x nG x samples x nJ. If False, returns the best one only, so the resulting tensor has a shape nR x nG x nJ.
    :param ik_cost_function: The cost function to use for the inverse kinematics problem if return_all is False.
    :param batch_size: The batch size to use for the forward pass of the model.
    :param num_processes_get_q: The number of processes (cpu only) to use for the get_q function.
    :return: See return_all for info.
    """
    device = kinematic_chains.device
    model = get_model().to(device)

    assert len(kinematic_chains.shape) == 4, f'Expected 4D tensor, got {kinematic_chains.shape}'
    nR, nJ, _, _ = kinematic_chains.shape
    _, nG, _, _ = goals.shape
    eef = f'p{nJ}'

    t_zeros = {i: joint_transforms_to_t_zero(kinematic_chains[i], [f'p{j}' for j in range(1 + nJ)], se3type='numpy') for
               i in range(nR)}
    robots = {i: RobotRevolute({'num_joints': nJ, 'T_zero': t_zeros[i]}) for i in range(nR)}
    graphs = {i: ProblemGraphRevolute(robots[i]) for i in range(nR)}
    if return_all:
        q = torch.zeros((nR * nG, samples, nJ), device=device)
    else:
        q = torch.zeros((nR * nG, nJ), device=device)

    problems = list()
    for i, j in itertools.product(range(nR), range(nG)):
        graph = graphs[i]
        goal = goals[i, j]
        problems.append(generate_data_point_from_pose(graph, goal))

    goals = goals.detach().cpu().numpy()
    # FIXME: Create one data point per sample until forward_eval works correctly with more than one sample
    problems_times_samples = list(itertools.chain.from_iterable(zip(*[problems] * samples)))
    data = create_dataset_from_data_points(problems_times_samples)
    batch_size_forward = batch_size * samples
    loader = DataLoader(data, batch_size=batch_size_forward, shuffle=False, num_workers=0)

    for i, problem in enumerate(loader):
        problem = model.preprocess(problem)
        b = len(problem)  # The actual batch size (might be smaller than batch_size_forward at the end of the dataset)
        num_nodes_per_graph = int(problem.num_nodes / b)
        P_all_ = model.forward_eval(
            x=problem.pos,
            h=torch.cat((problem.type, problem.goal_data_repeated_per_node), dim=-1),
            edge_attr=problem.edge_attr,
            edge_attr_partial=problem.edge_attr_partial,
            edge_index=problem.edge_index_full,
            partial_goal_mask=problem.partial_goal_mask,
            nodes_per_single_graph=num_nodes_per_graph,
            batch_size=b,
            num_samples=1
        ).squeeze()
        # Rearrange, s.t. we have problem_nr x sample_nr x node_nr x 3
        P_all = P_all_.view(b // samples, samples, num_nodes_per_graph, 3)
        q_batch = _get_q_batch(P_all, graphs, goals, ik_cost_function, batch_size, return_all, i, nR, nG, num_processes_get_q)
        q[i * batch_size: (i + 1) * batch_size, ...] = q_batch

    return q.view(nR, nG, -1, nJ) if return_all else q.view(nR, nG, -1)
