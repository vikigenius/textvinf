#!/usr/bin/env python3
import torch
from pyro.distributions import Distribution


def monte_carlo_kl(q_z: Distribution, p_z: Distribution, z: torch.Tensor):
    # Compute the MC sample KL estimate with q_z posterior p_z prior and z ~ q_z
    batch_size = z.size(0)
    try:
        e_log_qz = -torch.sum(q_z.entropy()) / batch_size
    except NotImplementedError:
        e_log_qz = q_z.log_prob(z).mean()
    e_log_pz = p_z.log_prob(z).mean()
    return e_log_qz - e_log_pz


def flow_transform_kl(q_z0: Distribution, p_z: Distribution, z: torch.Tensor, logj: torch.Tensor):
    # Compute the MC sample KL estimate with q_z posterior p_z prior and z ~ q_z
    e_log_qz0 = - q_z0.entropy().mean()  # Alternative to q_zo.log_prob(z0)
    e_log_pz = p_z.log_prob(z).mean()
    return e_log_qz0 - e_log_pz - logj.mean()  # log_qz = log_qz0 - logj
