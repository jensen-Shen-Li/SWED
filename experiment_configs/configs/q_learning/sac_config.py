from lifelong_rl.models.networks import ParallelizedEnsembleFlattenMLP
from lifelong_rl.policies.base.base import MakeDeterministic
from lifelong_rl.policies.models.tanh_gaussian_policy import TanhGaussianPolicy
from lifelong_rl.trainers.q_learning.sac import SACTrainer
import lifelong_rl.util.pythonplusplus as ppp
import os
import torch
import lifelong_rl.torch.pytorch_util as ptu
from torch.nn import functional as F
import numpy as np


def get_probabilistic_num_min(num_mins):
    # allows the number of min to be a float
    floored_num_mins = np.floor(num_mins)
    if num_mins - floored_num_mins > 0.001:
        prob_for_higher_value = num_mins - floored_num_mins
        if np.random.uniform(0, 1) < prob_for_higher_value:
            return int(floored_num_mins+1)
        else:
            return int(floored_num_mins)
    else:
        return num_mins


def circular_sliding_window(x: int, num: int):
    data = np.arange(x)
    step = x + 1 - num
    start = 0
    while True:
        indices = [(start + i) % x for i in range(num)]
        yield data[indices]
        start = (start + step) % x


def get_config(
        variant,
        expl_env,
        eval_env,
        obs_dim,
        action_dim,
        replay_buffer,
):
    """
    Policy construction
    """

    num_qs = variant['trainer_kwargs']['num_qs']
    M = variant['policy_kwargs']['layer_size']
    num_q_layers = variant['policy_kwargs']['num_q_layers']
    num_p_layers = variant['policy_kwargs']['num_p_layers']

    num_mins_to_use = get_probabilistic_num_min(variant['trainer_kwargs']['q_samples'])
    # sample_idxs = np.random.choice(num_qs, num_mins_to_use, replace=False)

    sample_idxs = next(circular_sliding_window(num_qs, num_mins_to_use))

    all_nums = np.arange(num_qs)
    rest_idx = np.setdiff1d(all_nums, sample_idxs)

    qfs, target_qfs = ppp.group_init(
        2,
        ParallelizedEnsembleFlattenMLP,
        ensemble_size=num_qs,
        ensemble_idx=sample_idxs,
        hidden_sizes=[M] * num_q_layers,
        input_size=obs_dim + action_dim,
        output_size=1,
        layer_norm=None,
    )

    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M] * num_p_layers,
        layer_norm=None,
    )

    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qfs=qfs,
        target_qfs=target_qfs,
        replay_buffer=replay_buffer,
        **variant['trainer_kwargs'],
    )
    """
    Create config dict
    """

    config = dict()
    config.update(
        dict(
            trainer=trainer,
            exploration_policy=policy,
            evaluation_policy=MakeDeterministic(policy),
            exploration_env=expl_env,
            evaluation_env=eval_env,
            replay_buffer=replay_buffer,
            qfs=qfs,
        ))
    config['algorithm_kwargs'] = variant.get('algorithm_kwargs', dict())

    return config
