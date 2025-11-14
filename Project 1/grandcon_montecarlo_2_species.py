from __future__ import annotations
import numpy as np
from typing import Dict, Tuple
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class Params:
    epsilon_A: float
    epsilon_B: float
    epsilon_AA: float
    epsilon_BB: float
    epsilon_AB: float

    mu_A: float
    mu_B: float

    T: float
def initialize_lattice(size: int) -> np.ndarray:
    return np.zeros((size, size), dtype=np.int8)

def compute_neighbor_indices(size: int) -> dict[Tuple[int, int], Tuple[Tuple[int, int],...]]:
    neighbors: Dict[Tuple[int, int], Tuple[Tuple[int, int],...]] = {}
    for i in range(size):
        for j in range(size):
            neighbors[(i, j)] = (
                ((i - 1) % size, j),
                ((i + 1) % size, j),
                (i, (j - 1) % size),
                (i, (j + 1) % size),
            )
    return neighbors
def calculate_interaction_energy(
    lattice: np.ndarray,
    site: Tuple[int, int],
    particle: int,
    neighbor_indices: Dict[Tuple[int, int], Tuple[Tuple[int, int],...]],
    eps_AA: float,
    eps_BB: float,
    eps_AB: float,
) -> float:
    i, j = site
    energy = 0.0
    for ni, nj in neighbor_indices[(i, j)]:
        s_nb = lattice[ni, nj]

        if s_nb == 0:
            continue
        # particle: 1 = A, 2 = B ; s_nb: neighbor species
        if particle == 1:
            if s_nb == 1:
                energy += eps_AA
            else:
                energy += eps_AB
        else:
            if s_nb == 2:
                energy += eps_BB
            else:
                energy += eps_AB
    return energy
def delta_E_for_add(lattice: np.ndarray, site: Tuple[int, int], particle: int, neighbor_indices: Dict[Tuple[int, int], Tuple[Tuple[int, int], ...]], p: Params) -> float:
    if particle == 1:
        eps_site = p.epsilon_A
        e_pairs = calculate_interaction_energy(
            lattice, site, 1, neighbor_indices, p.epsilon_AA, p.epsilon_BB, p.epsilon_AB
        )
    else:
        eps_site = p.epsilon_B
        e_pairs = calculate_interaction_energy(
            lattice, site, 2, neighbor_indices, p.epsilon_AA, p.epsilon_BB, p.epsilon_AB
        )
    return eps_site + e_pairs


def delta_E_for_remove(lattice: np.ndarray, site: Tuple[int, int], neighbor_indices: Dict[Tuple[int, int], Tuple[Tuple[int, int],...]], p: Params) -> float:
    i, j = site
    particle= lattice[i, j]
    if particle == 0:
        # nothing to remove
        return 0.0
    if particle == 1:
        eps_site = p.epsilon_A
        e_pairs = calculate_interaction_energy(
            lattice, site, 1, neighbor_indices, p.epsilon_AA, p.epsilon_BB, p.epsilon_AB
        )
    else:
        eps_site = p.epsilon_B
        e_pairs = calculate_interaction_energy(
            lattice, site, 2, neighbor_indices, p.epsilon_AA, p.epsilon_BB, p.epsilon_AB
        )
    # removing a particle removes its site energy and its pair interactions
    return - (eps_site + e_pairs)
def attempt_move(lattice: np.ndarray, N_A: int, N_B: int, N_empty: int, neighbor_indices: Dict[Tuple[int, int], Tuple[Tuple[int, int],...]], p: Params, rng: np.random.Generator,) -> tuple[int, int, int]:     
    size = lattice.shape[0]
    N_sites = size * size
    beta = 1.0 / p.T
    u = rng.random()
    if u < 1.0/3.0:
        if N_empty == 0:
            return N_A, N_B, N_empty
        while True:
            i = rng.integers(0, size)
            j = rng.integers(0, size)
            if lattice[i, j] == 0:
                site = (i, j); break
        
        particle = 1
        mu = float(p.mu_A)
        N_s = N_A
        dE = delta_E_for_add(lattice, site, particle, neighbor_indices, p)
        acc = min(1.0, (N_empty / (N_s + 1.0)) * np.exp(-beta * (dE - mu)))
        if rng.random() < acc:
            lattice[site] = particle
            N_A += 1
            N_empty -= 1
    elif u < 2.0/3.0:
        if N_empty == 0:
            return N_A, N_B, N_empty
        while True:
            i = rng.integers(0, size)
            j = rng.integers(0, size)
            if lattice[i, j] == 0:
                site = (i, j); break
        particle = 2
        mu = float(p.mu_B)
        N_s = N_B
        dE = delta_E_for_add(lattice, site, particle, neighbor_indices, p)
        acc = min(1.0, (N_empty / (N_s + 1.0)) * np.exp(-beta * (dE - mu)))
        if rng.random() < acc:
            lattice[site] = particle
            N_B += 1
            N_empty -= 1
    else:
        if N_empty == N_sites:
            return N_A, N_B, N_empty
        while True:
            i = rng.integers(0, size)
            j = rng.integers(0, size)
            if lattice[i, j] != 0:
                site = (i, j); break
        particle = lattice[site]
        if particle == 1:
                mu, N_s = p.mu_A, N_A
        else:
                mu, N_s = p.mu_B, N_B
        dE = delta_E_for_remove(lattice, site, neighbor_indices, p)
        acc = min(1.0, (N_s / (N_empty + 1.0)) * np.exp(-beta * (dE + mu)))
        if rng.random() < acc:
            lattice[site] = 0
            if particle == 1:
                N_A -= 1
            else:
                N_B -= 1
            N_empty += 1
    return N_A, N_B, N_empty
      
def run_simulation(size: int, n_steps: int, p: Params, seed: int =42):
    rng = np.random.default_rng(seed)
    lattice = initialize_lattice(size)
    neighbor_indices = compute_neighbor_indices(size)
    N_sites = size * size
    N_A = 0
    N_B = 0
    N_empty = N_sites
    coverage_A = np.zeros(n_steps)
    coverage_B = np.zeros(n_steps)
    for step in range(n_steps):
        N_A, N_B, N_empty = attempt_move(
            lattice, N_A, N_B, N_empty, neighbor_indices, p, rng
        )
        coverage_A[step] = N_A / N_sites
        coverage_B[step] = N_B / N_sites
    return lattice, coverage_A, coverage_B
def plot_lattice(lattice: np.ndarray, title : str= ""):
    size = lattice.shape[0]
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks(np.arange(0, size + 1), minor=True)
    ax.set_yticks(np.arange(0, size + 1), minor=True)
    ax.grid(which="minor", linewidth = .5, color= "black")
    xsA, ysA, xsB, ysB = [], [], [], []
    for i in range(size):
        for j in range(size):
            if lattice[i, j] == 1:
                xsA.append(i + 0.5)
                ysA.append(j + 0.5)
            elif lattice[i, j] == 2:
                xsB.append(i + 0.5)
                ysB.append(j + 0.5)
    ax.scatter(xsA, ysA, s=40, label = "A")
    ax.scatter(xsB, ysB, s =40, label = "B")
    ax.set_title(title)
    ax.legend()
    plt.show()

if __name__ == "__main__":
    p = Params(
        epsilon_A=-0.1,
        epsilon_B=-0.1,
        epsilon_AA=0.0,
        epsilon_BB=0.0,
        epsilon_AB=0.0,
        mu_A=-.1,
        mu_B=-.1,
        T=.01,
    )
    lattice, covA, covB = run_simulation(size=4, n_steps=100000, p=p, seed=123)
    print("Final Lattice:")
    plot_lattice(lattice, title="Final Lattice Configuration")  









