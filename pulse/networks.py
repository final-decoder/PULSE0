"""Benchmark network construction protocols (App. 'Network generation')."""
from __future__ import annotations

import numpy as np

from .model import HawkesNetwork, KernelBank


def _spectral_radius(K: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(K))))


def generate_network(seed: int, p: int, q: int, radius: float) -> HawkesNetwork:
    """Random stable network; entry (i, j) is excitation from j into i.

    Protocol (App. 'Network generation'):
    off-diagonal edges independently with probability 0.10, weights
    U[0.04, 0.28]; self-excitation U[0.06, 0.38]; each hidden population gets
    two distinct observed children with weights [0.12, 0.42]; with probability
    0.8 an observed-to-hidden input of weight [0.12, 0.38]; with probability
    0.6 a link from the next hidden population (circular) of weight
    [0.08, 0.24]; add 0.25 to one random observed entry; rescale to the
    target spectral radius. Edge kernels are Gamma with shape in {1, 2, 3} and
    mean delay log-uniform in [0.025, 0.22] s. Immigrant rates U[0.008, 0.025],
    observed rates with an extra factor U[0.65, 2.5].
    """
    rng = np.random.default_rng(seed)
    n = p + q
    K = np.zeros((n, n))
    off = ~np.eye(n, dtype=bool)
    active = off & (rng.random((n, n)) < 0.10)
    K[active] = rng.uniform(0.04, 0.28, size=int(active.sum()))
    K[np.diag_indices(n)] = rng.uniform(0.06, 0.38, size=n)
    for h in range(p, n):
        children = rng.choice(p, size=min(2, p), replace=False)
        K[children, h] += rng.uniform(0.12, 0.42, size=len(children))
        if rng.random() < 0.8:
            K[h, rng.integers(p)] += rng.uniform(0.12, 0.38)
        if q > 1 and rng.random() < 0.6:
            nxt = p + (h - p + 1) % q
            K[h, nxt] += rng.uniform(0.08, 0.24)
    K[rng.integers(p), rng.integers(p)] += 0.25
    K *= radius / _spectral_radius(K)

    active = K > 0
    shapes = rng.integers(1, 4, size=(n, n)).astype(float)
    means = np.exp(rng.uniform(np.log(0.025), np.log(0.22), size=(n, n)))
    kernels = KernelBank(
        weights=K,
        shapes=np.where(active, shapes, 1.0),
        means=np.where(active, means, 0.1),
    )
    mu = rng.uniform(0.008, 0.025, size=n)
    mu[:p] *= rng.uniform(0.65, 2.5, size=p)
    return HawkesNetwork(mu=mu, kernels=kernels, p=p)


def _cyclic_permutation(k: int) -> np.ndarray:
    P = np.zeros((k, k))
    P[np.arange(k), (np.arange(k) + 1) % k] = 1.0
    return P


def equivalence_family(f: int, p: int = 4):
    """Response-equivalent family f (App. 'Constructing and evaluating
    response-equivalent networks').

    Base network: seed 51000 + f, 4 observed / 1 hidden, spectral radius
    0.68 + 0.02 (f mod 4). The hidden population is replaced by k in
    {1, 2, 4} clones with S_k = 0.4 I + 0.6 P_k (cyclic), preserving (R, r).
    Variants use exponential edge kernels with mean delays 45, 90, 180 ms.
    Returns dict {k: HawkesNetwork}.
    """
    base = generate_network(seed=51000 + f, p=p, q=1,
                            radius=0.68 + 0.02 * (f % 4))
    A = base.K[:p, :p]
    B = base.K[:p, p:]
    C = base.K[p:, :p]
    h = float(base.K[p, p])
    variants = {}
    for k, mean_ms in ((1, 45.0), (2, 90.0), (4, 180.0)):
        S = 0.4 * np.eye(k) + 0.6 * _cyclic_permutation(k)
        K_k = np.block([[A, B @ np.ones((1, k)) / k],
                        [np.ones((k, 1)) @ C, h * S]])
        mu_k = np.concatenate([base.mu[:p], base.mu[p:] * np.ones(k)])
        active = K_k > 0
        kernels = KernelBank(
            weights=K_k,
            shapes=np.ones((p + k, p + k)),  # exponential kernels
            means=np.where(active, mean_ms / 1000.0, 0.1),
        )
        variants[k] = HawkesNetwork(mu=mu_k, kernels=kernels, p=p)
    return variants


def contaminated_response(network, epsilon: float) -> np.ndarray:
    """R_contaminated = R + eps Q_OH (1_q / q) 1_p^T (App. 'Direct recruitment'),
    with Q = (I - K)^{-1} the complete integrated resolvent."""
    p, q = network.p, network.q
    Q = np.linalg.inv(np.eye(network.n) - network.K)
    R = Q[:p, :p].copy()
    if q > 0 and epsilon > 0.0:
        R += epsilon * (Q[:p, p:] @ np.ones((q, p)) / q)
    return R


def timing_network(participant_index: int, n1_latencies_ms,
                   p: int = 8, q: int = 8, radius: float = 0.76) -> HawkesNetwork:
    """Empirical-timing benchmark network (App. 'timing'): seed 31000 + u,
    standard connectivity/baseline distributions, Gamma shapes in {1, 2, 3},
    and mean delays drawn with replacement from the participant's empirical
    N1 latency distribution (milliseconds)."""
    seed = 31000 + int(participant_index)
    net = generate_network(seed=seed, p=p, q=q, radius=radius)
    rng = np.random.default_rng(seed)
    lat = np.asarray(n1_latencies_ms, dtype=float)
    active = net.K > 0
    means = np.full((p + q, p + q), 0.1)
    means[active] = rng.choice(lat, size=int(active.sum()), replace=True) / 1000.0
    kernels = KernelBank(net.K, net.kernels.shapes, means)
    return HawkesNetwork(mu=net.mu, kernels=kernels, p=p)
