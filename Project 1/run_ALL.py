from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from grandcon_montecarlo_2_species import Params, run_simulation
kB_eV_per_K = 8.617333262145e-5  # Boltzmann constant in eV/K
ALL_SCENARIOS = { "Ideal": dict(epsilon_A=-.10, epsilon_B=-.10, epsilon_AA=0.0, epsilon_BB=0.0, epsilon_AB=0.0),
                  "Repulsive": dict(epsilon_A=-.10, epsilon_B= -.10, epsilon_AA=0.05, epsilon_BB=0.05, epsilon_AB=0.05),
                  "Attractive": dict(epsilon_A=-.10, epsilon_B= -.10, epsilon_AA=-0.05, epsilon_BB=-0.05, epsilon_AB=-0.05),
                "Immiscible": dict(epsilon_A=-.10, epsilon_B= -.10, epsilon_AA=-0.05, epsilon_BB=-0.05, epsilon_AB=0.05),
                "LikeUnlike": dict(epsilon_A=-.10, epsilon_B= -.10, epsilon_AA=0.05, epsilon_BB=0.05, epsilon_AB=-0.05),}
def run_mus_A_T_grid(scenario_name: str, mu_H_list, Ts, mu_N_fixed=-.10, L=8, n_steps=120_000, burn_in=40_000, sample_last=20_000, seed=42, outdir= "results/figures"):
    cfg = ALL_SCENARIOS[scenario_name]
    Path(outdir).mkdir(parents=True, exist_ok=True)
    mus_A = np.array(mu_H_list, dtype= float)
    T_K = np.array(Ts, dtype= float)
    T_eV = kB_eV_per_K * T_K
    thetaN = np.zeros((len(mus_A), len(T_K)))
    thetaH = np.zeros((len(mus_A), len(T_K)))

    rng = np.random.default_rng(seed)
    for i, mu_B in enumerate(mus_A):
        mu_B_scalar = float(mu_B)
        for j, Tval in enumerate(T_eV):
            T_scalar = float(Tval)
            p = Params (epsilon_A=cfg["epsilon_A"],
                       epsilon_B=cfg["epsilon_B"],
                       epsilon_AA=cfg["epsilon_AA"],
                       epsilon_BB=cfg["epsilon_BB"],
                       epsilon_AB=cfg["epsilon_AB"],
                       mu_A = float(mu_N_fixed),
                       mu_B = mu_B_scalar,
                       T = T_scalar,
                       )
            lat, covA, covB = run_simulation(size=L,n_steps= n_steps, p=p, seed = int(rng.integers(0, 2**31 -1)))

            seriesA= covA[min(burn_in, len(covA)):]
            seriesB= covB[min(burn_in, len(covB)):]
            if len(seriesA) >= sample_last:
                seriesA= seriesA[-sample_last:]
                seriesB= seriesB[-sample_last:]
            thetaN[i, j]= float(np.mean(seriesA)) if len(seriesA) else float(covA[-1])
            thetaH[i, j]= float(np.mean(seriesB)) if len(seriesB) else float(covB[-1])

    outbase = Path(outdir) / f"{scenario_name}_grid"
    np.savez_compressed(f"{outbase}.npz", mu_H=mus_A, T_K=T_K, theta_N=thetaN, theta_H=thetaH, theta_tot = (thetaN + thetaH), meta = dict(mu_N_fixed= mu_N_fixed, L=L, n_steps=n_steps, burn_in= burn_in, sample_last= sample_last),)
    print(f"[saved] {outbase}.npz")
    
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(9.0, 3.4))
    im0 = axs[0].pcolormesh(mus_A, T_K, thetaN.T, cmap = "viridis", vmin= 0, vmax=1)
    axs[0].set_title(r"$\langle \theta_N \rangle$"); axs[0].set_xlabel(r'$\mu_H$ (eV)'); axs[0].set_ylabel(r'$T$ (K)')
    
    im1 = axs[1].pcolormesh(mus_A, T_K, thetaH.T, cmap = "viridis", vmin= 0, vmax=1)
    axs[1].set_title(r"$\langle \theta_H\rangle$"); axs[1].set_xlabel(r'$\mu_H$ (eV)'); axs[1].set_yticks([])
    
    im2 = axs[2].pcolormesh(mus_A, T_K, (thetaN+ thetaH).T, cmap = "viridis", vmin= 0, vmax=1)
    axs[2].set_title(r"$\langle \theta \rangle$"); axs[2].set_xlabel(r'$\mu_H$ (eV)'); axs[2].set_yticks([])
    
    
    cbar = fig.colorbar(im2, ax=axs[2], fraction=.08); cbar.set_label(r'Coverage')
    fig.tight_layout()
    png_path = Path(outdir) / f"phase_{scenario_name}.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {png_path}")
def _plot_lattice_axes(ax, lat, title: str = "" ):
     L = lat.shape[0]
     ax.set_aspect("equal")
     ax.set_xlim(0, L)
     ax.set_ylim(0, L)
     ax.set_xticks([])
     ax.set_yticks([])
     ax.set_xticks(np.arange(0, L + 1), minor=True)
     ax.set_yticks(np.arange(0, L + 1), minor=True)
     ax.grid(which="minor", linewidth = .5, color= "black")    
     xsN, ysN, xsH, ysH = [], [], [], []
     for i in range(L):
            for j in range(L):
                if lat[i, j] == 1:
                    xsN.append(i + 0.5)
                    ysN.append(j + 0.5)
                elif lat[i, j] == 2:
                    xsH.append(i + 0.5)
                    ysH.append(j + 0.5)
     if xsN:
            ax.scatter(xsN, ysN, s=38, edgecolor = "none", label = "N", c = "#1976d2")
     if xsH:
            ax.scatter(xsH, ysH, s =38, edgecolor = "none", label = "H", c= "#ef5350")
     ax.set_title(title)
def generate_lattice_snapshots(scenario_name: str, points: list[tuple[float, float]], mu_N_fixed=-.10, L=8, n_steps=60_000, burn_in=40_000, seed=123, outdir= "results/snaps", ALL_SCENARIOS: dict | None = None):
    assert len(points) == 3, "points must be a list of three (mu_H, T_K) tuples"
    if ALL_SCENARIOS is None:
        cfg = globals()["ALL_SCENARIOS"][scenario_name]
    else:
        cfg = ALL_SCENARIOS[scenario_name]
    Path(outdir).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    lattices = []
    titles = []
    for idx, (mu_H, T_K) in enumerate(points):
        p = Params (epsilon_A=cfg["epsilon_A"],
                       epsilon_B=cfg["epsilon_B"], epsilon_AA=cfg["epsilon_AA"],
                       epsilon_BB=cfg["epsilon_BB"], epsilon_AB=cfg["epsilon_AB"],
                       mu_A = mu_N_fixed,
                       mu_B = mu_H,
                       T = kB_eV_per_K * T_K,
                       )
        run_seed = int(rng.integers(0, 2**31 -1))
        lat, covA, covB = run_simulation(size=L,n_steps= n_steps, p=p, seed = run_seed)

        lattices.append(lat.copy())
        T_int = int(np.round(T_K))
        titles.append(rf"$\mu_H={mu_H:.2f}$ eV, $T= {T_int}$ K")   
        np.save(Path(outdir) / f"lat_{scenario_name}_snap_{idx+1}__muH_{mu_H:.2f}__T_{int(T_K)}K.npy", lat)
    fig, axs = plt.subplots(1, 3, figsize=(9.2, 3.2))
    for k in range(3):
        _plot_lattice_axes(axs[k], lattices[k], titles[k])
    from matplotlib.lines import Line2D
    legend_handles = [Line2D([], [], marker = 'o', linestyle = '', markersize =8, color= "#1976d2", label="N"),                 
                      Line2D([], [], marker = 'o', linestyle = '', markersize =8, color= "#ef5350", label="H"),]
    fig.legend(legend_handles, ["N", "H"], loc="upper left", ncol=2, bbox_to_anchor=(0.06, 1.02))
    
    fig.suptitle(f"{scenario_name} - lattice snapshots", fontsize=12, y=1.05)
    fig.tight_layout()
    png_path = Path(outdir) / f"snaps_{scenario_name}.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {png_path}")
             
if __name__ == "__main__":
            mu_H_list = np.linspace(-0.20, 0.0, 7)
            Ts = np.linspace(25.0, 250, 7)
            for scen in ALL_SCENARIOS: 
                if scen in ("Attractive", "Immiscible"):
                    mu_H_list = np.linspace(-.80, 0.0, 13)
                    mu_N_fixed = -.10
                elif scen == "LikeUnlike":
                    mu_H_list = np.linspace(-.50, 0.0, 11)
                    mu_N_fixed = -.10
                else:
                    mu_H_list = np.linspace(-0.20, 0.0, 7)
                    mu_N_fixed = -.10
                       

                 
                run_mus_A_T_grid(scenario_name=scen, mu_H_list= mu_H_list, Ts= Ts, mu_N_fixed=mu_N_fixed, L=8, n_steps=120_000, burn_in=40_000, sample_last=20_000, seed=42, outdir="results/sweeps")
                T_show = .01/kB_eV_per_K

                points = [( -.2, T_show), (-.1, T_show), (0, T_show)]
                generate_lattice_snapshots(scenario_name=scen, points= points, mu_N_fixed= mu_N_fixed, L=8, n_steps=60_000, burn_in=40_000, seed=123, outdir="results/snaps", ALL_SCENARIOS= ALL_SCENARIOS)
