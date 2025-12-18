from __future__ import annotations
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
# I included all simmulation parameters here
@dataclass(frozen=True)
class Params:
    n: int = 20
    box_size: float = 100.0
    mass : float = 1.0
    k: float = 1.0
    r0: float = 1.0
    sigma: float = 1.0
    eps_rep: float = 1.0
    eps_att: float = .5
    dt: float = 0.0005
    total_steps: int = 10_000
    rescale_interval: int = 1
    kB: float = 1.0
    r_min_factor: float = 0.85
    temperatures: tuple[float, float, int] = (0.1, 1.0, 10)
    equil_fraction: float = 0.3
# this function will be used for our box model with periodic boundary conditions
def apply_pbc(x:np.ndarray, box_size:float) -> np.ndarray:
    return x% box_size
#function to make sure we find the shortest distance between points with our setup
def minimum_image(dr: np.ndarray, box_size: float) -> np.ndarray:
    dr = dr - box_size * np.round(dr / box_size)
    return dr
# adapt our chain setup to account for periodic boundary conditions
def unwrap_chain(x: np.ndarray, box_size: float) -> np.ndarray:
    xu = np.zeros_like(x)
    xu[0] = x[0]
    for i in range(1, x.shape[0]):
        dr = minimum_image(x[i] - x[i - 1], box_size)
        xu[i] = xu[i - 1] + dr
    return xu
#This function will give us the random unit vectors for our chain
def _random_unit_vectors(rng: np.random.Generator, n: int) -> np.ndarray:
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v
#main function for chain initialization
def initialize_chain(rng: np.random.Generator, n: int, box_size: float, r0: float, sigma: float, min_sep_factor: float =.9, max_tries: int = 5000) -> np.ndarray:
    x = np.zeros((n, 3), dtype= float)
    cur = np.array([box_size / 2, box_size / 2, box_size / 2], dtype= float)
    x[0] = cur
    # I set a minimum separation distance
    min_sep = min_sep_factor* sigma

    for i in range(1, n):
        for _ in range(max_tries):
            direction = _random_unit_vectors(rng, 1)[0]
            trial = apply_pbc(x[i - 1] + direction * r0, box_size)
            too_close = False
            for j in range(i-1):
                dr = minimum_image(trial - x[j], box_size)
                if np.linalg.norm(dr) < min_sep:
                    too_close = True
                    break
            if not too_close:
                x[i] = trial
                break
        else:
            raise RuntimeError(f"Failed to place bead {i} after {max_tries} tries.")
    return x
#function to initialize velocities based on target temperature
def initialize_velocities(rng: np.random.Generator, n: int, target_T: float, mass: float, kB: float) -> np.ndarray:
    v = rng.normal(scale= np.sqrt(kB * target_T / mass), size=(n, 3))
    v -= np.mean(v, axis=0, keepdims=True)
    
    return v
#We need a harmonic model for our bonded interactions
def harmonic_forces(x: np.ndarray, k: float, r0: float, box_size: float) -> np.ndarray:
    n = x.shape[0]
    f = np.zeros_like(x)
    for i in range(n - 1):
        dr = minimum_image(x[i + 1] - x[i], box_size)
        r = np.linalg.norm(dr)
        if r == 0.0:
            continue
        fij = (-k * (r - r0)) * (dr / r)
        f[i] -= fij
        f[i + 1] += fij
    return f
#We use a lennard-jones potential for non-bonded interactions

# use the attractive part of the LJ potential for longer distances
def _lj_force_energy(r: float, eps: float, sig: float) -> tuple[float, float]:
    inv = sig / r
    inv6 = inv ** 6
    inv12 = inv6 ** 2
    U = 4.0 * eps * (inv12 - inv6)
    Fmag = 24.0 * eps * (2*inv12 - inv6) / r
    return U, Fmag
def nonbonded_forces_and_energy(x: np.ndarray, eps_rep, eps_att, sig, box_size, r_min_factor=0.85) -> tuple[np.ndarray, float]:
    rcut_rep = (2.0 ** (1.0 / 6.0)) * sig
    rcut_att = 2.5 * sig

    n = x.shape[0]
    f = np.zeros_like(x)
    U= 0.0
    for i in range(n - 1):
        for j in range(i + 1, n):
            sep = j - i
            if sep == 1:
                continue
            dr = minimum_image(x[j] - x[i], box_size)
            r_raw = np.linalg.norm(dr)
            r_min = r_min_factor * sig
            
            if r_raw < 1e-12:
                continue
            r_hat = dr / r_raw
            r_eff = max(r_raw, r_min_factor*sig)
            if sep == 2:
                if r_eff>= rcut_rep:
                   continue
                Uij, Fmag = _lj_force_energy(r_eff, eps_rep, sig)
            elif sep > 2:
                if r_eff>= rcut_att:
                  continue
                Uij, Fmag = _lj_force_energy(r_eff, eps_att, sig)
            else:
                continue
            fij = Fmag * r_hat
            f[i] -= fij
            f[j] += fij
            U += Uij
    return f, U
def total_forces_and_energy(x: np.ndarray, p: Params) -> tuple[np.ndarray, float]:
   f_b = harmonic_forces(x, p.k, p.r0, p.box_size)
   f_nb, U_nb = nonbonded_forces_and_energy(x, p.eps_rep, p.eps_att, p.sigma, p.box_size, r_min_factor= 0.85)
   return f_b + f_nb, U_nb

def velocity_verlet(x: np.ndarray, v: np.ndarray, f: np.ndarray, dt: float, mass: float, box_size: float, force_fn) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    v_half = v + 0.5 * (f / mass) * dt
    x_new = apply_pbc(x + v_half * dt, box_size)
    f_new, U_new = force_fn(x_new)
    v_new = v_half + 0.5 * (f_new / mass) * dt
    return x_new, v_new, f_new, U_new
def kinetic_energy(v: np.ndarray, mass: float) -> float:
    return 0.5 * mass * float(np.sum(v ** 2 ))

def instantaneous_temperature(K: float, n: int, kB: float) -> float:
     return (2.0 * K) / (3.0 * n * kB)
def rescale_velocities(v: np.ndarray, target_T: float, mass: float, kB: float) -> np.ndarray:
  K = kinetic_energy(v, mass)
  T = instantaneous_temperature(K, v.shape[0], kB)
  if T<= 0.0:
      return v
  return v * np.sqrt(target_T / T)
def radius_of_gyration(x: np.ndarray) -> float:
    x_cm = x.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((x - x_cm) ** 2, axis=1))))
def end_to_end_distance(x: np.ndarray, box_size: float) -> float:
    dr = minimum_image(x[-1] - x[0], box_size)
    return float(np.linalg.norm(dr))
def run_md(rng: np.random.Generator, p: Params, target_T: float) -> dict[str, np.ndarray | float]:
    x = initialize_chain(rng, p.n, p.box_size, p.r0, p.sigma)
    v = initialize_velocities(rng, p.n, target_T, p.mass, p.kB)
    
    

    def force_fn(x_):
        return total_forces_and_energy(x_, p)
    f, U = force_fn(x)
    rg = np.zeros(p.total_steps, dtype= float)
    ree = np.zeros(p.total_steps, dtype= float)
    U_nb = np.zeros(p.total_steps, dtype= float)
    T_inst = np.zeros(p.total_steps, dtype= float)
    for step in range(p.total_steps):
        x, v, f, U = velocity_verlet(x, v, f, p.dt, p.mass, p.box_size, force_fn)
        if step % p.rescale_interval == 0:
            v = rescale_velocities(v, target_T, p.mass, p.kB)
        xu= unwrap_chain(x, p.box_size)
        rg[step] = radius_of_gyration(xu)
        ree[step] = float(np.linalg.norm(xu[-1] - xu[0]))
        U_nb[step] = U
        T_inst[step] = instantaneous_temperature(kinetic_energy(v, p.mass), p.n, p.kB)
    return {"x_final": x, "v_final": v, "Rg_trace": rg, "Ree_trace": ree, "U_nb": U_nb, "T_inst": T_inst}
def sweep_temperatures(p: Params, seed: int = 0):
    rng = np.random.default_rng(seed)
    Tmin, Tmax, nT = p.temperatures
    Ts= np.linspace(Tmin, Tmax, nT)
    Rg_mean = np.zeros_like(Ts)
    Ree_mean = np.zeros_like(Ts)
    U_mean = np.zeros_like(Ts)
    T_mean = np.zeros_like(Ts)

    x_finals = []
    eq_start = int(np.floor(p.equil_fraction* p.total_steps))
    for idx, T in enumerate(Ts):
        out = run_md(rng, p, float(T))
        Rg_mean[idx] =out["Rg_trace"][eq_start:].mean()
        Ree_mean[idx] = out["Ree_trace"][eq_start:].mean()
        U_mean[idx] = out["U_nb"][eq_start:].mean()
        T_mean[idx] = out["T_inst"][eq_start:].mean()
        x_finals.append(out["x_final"])
        print(f"T={T:.3f}: <Rg>={Rg_mean[idx]:.3f}, <Ree>={Ree_mean[idx]:.3f}, <U_nb>={U_mean[idx]:.3f}, <T>={T_mean[idx]:.3f}")
    return Ts, Rg_mean, Ree_mean, U_mean, T_mean, x_finals

def plot_sweep(Ts, Rg, Ree, U, T_star= None):
    plt.figure()
    plt.plot(Ts, Rg, marker = "o")
    if T_star is not None:
        plt.axvline(T_star, color='red', linestyle='--')
       
    plt.xlabel("Temperature")
    plt.ylabel("Radius of Gyration, Rg")
    plt.title("Rg vs Temperature")
    plt.figure()
    plt.plot(Ts, Ree, marker = "o")
    if T_star is not None:
        plt.axvline(T_star, linestyle='--')
    plt.xlabel("Temperature")
    plt.ylabel("End-to-End Distance, Ree")
    plt.title("Ree vs Temperature")
    plt.figure()
    plt.plot(Ts, U, marker = "o")
    if T_star is not None:
        plt.axvline(T_star, linestyle='--')
    plt.xlabel("Temperature")
    plt.ylabel("Average Non-bonded PotentialEnergy, U_nb")
    plt.title("U_nb vs Temperature")
    ax= plt.gca()
    ax.ticklabel_format(axis= "y", style = "plain", useOffset= False)
    plt.show()
def plot_chain_3d(x: np.ndarray, box_size: float, title: str = "Polymer Configuration"):
    xu = np.zeros_like(x)
    xu[0]= x[0]
    for i in range(1, x.shape[0]):
        dr = minimum_image(x[i] - x[i - 1], box_size)
        xu[i] = xu[i - 1] + dr
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xu[:, 0], xu[:, 1], xu[:, 2], marker='o')
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()
if __name__ == "__main__":
    p = Params(n = 20, box_size=100.0, mass=1.0, k=1.0, r0=1.0, sigma=1.0, eps_rep=1.0, eps_att=0.5, dt=0.0005, total_steps=8000, rescale_interval=1, temperatures=(0.1, 1.0, 10), equil_fraction=0.3)
    Ts, Rg_mean, Ree_mean, U_mean, T_mean, x_finals = sweep_temperatures(p, seed=42)
    
    dRg_dT = np.gradient(Rg_mean, Ts)
    dRee_dT = np.gradient(Ree_mean, Ts)
    i_star_rg = int(np.argmax(np.abs(dRg_dT)))
    i_star_ree = int(np.argmax(np.abs(dRee_dT)))
    T_star_rg = float(Ts[i_star_rg])
    T_star_ree= float(Ts[i_star_ree])
    T_star= T_star_rg
    print(f"T* from max |d<Rg>/dT|: {T_star_rg:.3f}")
    print(f"T* from max |d<Ree>/dT|: {T_star_ree:.3f}")
    print(f"Using T* = {T_star:.3f} for vertical plot line")
    plot_sweep(Ts, Rg_mean, Ree_mean, U_mean, T_star=T_star)

    plot_chain_3d(x_finals[0], p.box_size, title=f"Final Configuration at T={Ts[0]:.2f}")
    idx = int(np.argmin(np.abs(Ts-.5)))
    plot_chain_3d(x_finals[idx], p.box_size, title=f"Final Configuration at T={Ts[idx]:.2f}")
    plot_chain_3d(x_finals[-1], p.box_size, title=f"Final Configuration at T={Ts[-1]:.2f}")