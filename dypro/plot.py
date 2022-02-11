from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rich.progress import track
from .dynamic import BaseChart
from .pci import functional as F
from .config import Parameters, AdjConf, PlotConf

plt.style.use(["science", "ieee"])


@dataclass
class PlotGraph:
    chart: BaseChart
    proposed_df: pd.DataFrame
    param: Parameters
    adj_conf: AdjConf
    plot_conf: PlotConf
    bothe_k1: np.ndarray
    pearn_k2: np.ndarray
    figname: str

    def plot_k2(self, k1_list: list[str], save_path):
        max_len = len(self.plot_conf.k2_df[k1_list[-1]].dropna())
        with plt.style.context(["science", "ieee"]):
            fig, ax = plt.subplots()
            plt_param = dict(xlabel="$n$", ylabel="$k_2$")
            for k1 in k1_list:
                k2 = self.plot_conf.k2_df[k1]
                ax.plot(
                    np.arange(2, max_len + 2),
                    k2[:max_len],
                    label=f"$k_1$={float(k1):.1f}",
                )
            ax.legend(loc="upper right")
            ax.autoscale(tight=True)
            ax.set(**plt_param)
            fig.savefig(save_path, dpi=self.plot_conf.dpi, bbox_inches="tight")
            plt.close()

    def cpk(self, save_path="./cpk_comparison.png"):
        """Comparing dynamic cpk brtween proposed and previous method."""
        mean, sigma, USL, LSL = (
            self.param.mean,
            self.param.sigma,
            self.param.USL,
            self.param.LSL,
        )
        k1 = self.proposed_df["k1 min"].values
        k2 = self.proposed_df["k2 min"].values

        with plt.style.context(["science", "ieee"]):
            fig, ax = plt.subplots()
            plt_param = dict(xlabel="$n$", ylabel="$Dynamic$ $C_{pk}$")
            ax.plot(
                self.adj_conf.n,
                F.dynamic_cpk(mean, sigma, USL, LSL, k1, k2),
                label="Proposed Method",
            )
            # ax.plot(
            #     self.adj_conf.n,
            #     F.dynamic_cpk(mean, sigma, USL, LSL, self.bothe_k1, 1),
            #     label="Mean Shift",
            # )
            # ax.plot(
            #     self.adj_conf.n,
            #     F.dynamic_cpk(mean, sigma, USL, LSL, 0, self.pearn_k2),
            #     label="Variance Change",
            # )
            ax.plot(
                self.adj_conf.n,
                F.dynamic_cpk(mean, sigma, USL, LSL, self.bothe_k1, self.pearn_k2),
                label="Previous Method",
            )

            ax.legend(loc="lower right")
            ax.autoscale(tight=True)
            ax.set(**plt_param)
            fig.savefig(save_path, dpi=self.plot_conf.dpi, bbox_inches="tight")
            plt.close()

    def ncppm(self, save_path="ncppm_comarison.png"):
        mean, sigma, USL, LSL = (
            self.param.mean,
            self.param.sigma,
            self.param.USL,
            self.param.LSL,
        )
        k1 = self.proposed_df["k1 min"].values
        k2 = self.proposed_df["k2 min"].values

        with plt.style.context(["science", "ieee"]):
            fig, ax = plt.subplots()
            plt_param = dict(xlabel="$n$", ylabel="$ncppm$")
            ax.plot(
                self.adj_conf.n,
                F.ncppm(F.dynamic_cpk(mean, sigma, USL, LSL, k1, k2)),
                label="Proposed Method",
            )
            # ax.plot(
            #     self.adj_conf.n,
            #     F.ncppm(F.dynamic_cpk(mean, sigma, USL, LSL, self.bothe_k1, 1)),
            #     label="Mean Shift",
            # )
            # ax.plot(
            #     self.adj_conf.n,
            #     F.ncppm(F.dynamic_cpk(mean, sigma, USL, LSL, 0, self.pearn_k2)),
            #     label="Variance Change",
            # )
            ax.plot(
                self.adj_conf.n,
                F.ncppm(
                    F.dynamic_cpk(mean, sigma, USL, LSL, self.bothe_k1, self.pearn_k2)
                ),
                label="Previous Method",
            )

            ax.legend(loc="upper right")
            ax.autoscale(tight=True)
            ax.set(**plt_param)
            fig.savefig(save_path, dpi=self.plot_conf.dpi, bbox_inches="tight")
            plt.close()

    def k1_power(self, subgroup_size: list[int], save_path, k1_max=3):
        k1 = np.arange(0, k1_max + 0.01, 0.01)
        plt_param = dict(xlabel="$k_1$", ylabel="$power$", ylim=[0, 1])

        with plt.style.context(["science", "ieee"]):
            fig, ax = plt.subplots()
            for n in subgroup_size:
                ax.plot(k1, self.chart.power(k1, 1, n), label=f"n={n}")
            ax.legend(loc="lower right", title="Sample Size")
            ax.autoscale(tight=True)
            ax.set(**plt_param)
            fig.savefig(save_path, dpi=self.plot_conf.dpi, bbox_inches="tight")
            plt.close()

    def k2_power(self, subgroup_size: list[int], save_path, k2_max=3):
        k2 = np.arange(1, k2_max + 0.01, 0.01)
        plt_param = dict(xlabel="$k_2$", ylabel="$power$", ylim=[0, 1])

        with plt.style.context(["science", "ieee"]):
            fig, ax = plt.subplots()
            for n in subgroup_size:
                ax.plot(
                    k2,
                    np.array([self.chart.power(0, k2_, n) for k2_ in k2]),
                    label=f"n={n}",
                )
            ax.legend(loc="lower right", title="Sample Size")
            ax.autoscale(tight=True)
            ax.set(**plt_param)
            fig.savefig(save_path, dpi=self.plot_conf.dpi, bbox_inches="tight")
            plt.close()

    def k1_k2_power(
        self,
        n: int,
        save_path: str,
        k1_max: float = 3,
        k2_max: float = 3,
    ):
        """k1 and k2 vs power graph"""
        k1 = np.arange(0, k1_max + 0.01, 0.01)
        k2 = np.arange(1, k2_max + 0.01, 0.01)
        plt_param = dict(xlabel="$k_1$ and $k_2$", ylabel="$power$", ylim=[0, 1])

        with plt.style.context(["science", "ieee"]):
            fig, ax = plt.subplots()
            ax.plot(k1, self.chart.power(k1, 1, n), label=f"$k_1$")
            ax.plot(
                k2,
                np.array([self.chart.power(0, k2_, n) for k2_ in k2]),
                label=f"$k_2$",
            )
            ax.legend(loc="upper left")
            ax.autoscale(tight=True)
            ax.set(**plt_param)
            fig.savefig(save_path, dpi=self.plot_conf.dpi, bbox_inches="tight")
            plt.close()

    def plot_power_surface(
        self,
        save_path="./power_surface.png",
        n=5,
        k1_max=2,
        k2_max=3,
        step=0.01,
        add_power_line=False,
        alpha=1,
    ):
        """Plot detection power surface depends on pari (k1, k2)."""
        assert k2_max >= 1, "k2 maximize value must be greater than 1."
        k1 = np.arange(-k1_max, k1_max + step, step)
        k2 = np.arange(step, k2_max + step, step)
        k1, k2 = np.meshgrid(k1, k2)
        power = np.array(
            [
                self.chart.power(_k1, _k2, n)
                for _k1, _k2, in track(
                    zip(k1.flatten(), k2.flatten()),
                    total=len(k1.flatten()),
                    description=f"Plotting chart {n=} detection power surface...",
                )
            ]
        ).reshape(k1.shape)
        fig, ax = plt.subplots(
            figsize=self.plot_conf.figsize, subplot_kw={"projection": "3d"}
        )
        surf = ax.plot_surface(
            k1, k2, power, rstride=15, cstride=15, cmap="viridis", alpha=alpha
        )
        ax.view_init(azim=10)
        ax.set_xlabel("$k_1$", color="b")
        ax.set_ylabel("$k_2$", color="b")
        ax.set_zlabel("$power$", color="r")
        ax.set_xlim(-k1_max, k1_max)
        ax.set_ylim(0, k2_max + step)
        # fig.colorbar(surf, shrink=0.5, aspect=5)

        if add_power_line:
            k1_right = np.array(
                [float(k1) for k1 in self.plot_conf.k2_df.loc[n - 2, :].index]
            )
            k2_right = self.plot_conf.k2_df.loc[n - 2, :].values
            k1_left = -1 * np.flip(k1_right)
            k2_left = np.flip(k2_right)
            k1 = np.append(k1_left, k1_right)
            k2 = np.append(k2_left, k2_right)
            ax.plot(k1, k2, self.chart.power(k1, k2, n), c="r", ls="--", alpha=1)

        plt.savefig(save_path, dpi=self.plot_conf.dpi, bbox_inches="tight")
        plt.close()

    def plot_cpk_surface(
        self,
        save_path="./cpk_surface.png",
        k1_max=2,
        k2_max=3,
        step=0.01,
        add_power_line=False,
        alpha=1,
        n: Optional[int] = 30,
    ):
        """Plot 3D surface for cpk."""

        assert k1_max | k2_max > 0, "k1_stop and k2_stop must > 0"

        k1 = np.arange(0, k1_max + step, step)
        k2 = np.arange(1, k2_max + step, step)
        k1, k2 = np.meshgrid(k1, k2)
        dynamic_cpk = F.dynamic_cpk(
            mean=self.param.mean,
            sigma=self.param.sigma,
            USL=self.param.USL,
            LSL=self.param.LSL,
            k1=k1,
            k2=k2,
        )

        fig, ax = plt.subplots(
            figsize=self.plot_conf.figsize, subplot_kw={"projection": "3d"}
        )

        surf = ax.plot_surface(
            k1, k2, dynamic_cpk, rstride=15, cstride=15, cmap="viridis", alpha=alpha
        )

        if add_power_line:
            k1_proposed = np.array(list(map(float, self.plot_conf.k2_df.columns)))
            k2_proposed = self.plot_conf.k2_df.values[n - 2]
            idxs = np.where(k2_proposed == np.nan)
            k1_proposed = np.delete(k1_proposed, idxs)
            k2_proposed = np.delete(k2_proposed, idxs)
            cpk_proposed = F.dynamic_cpk(
                mean=self.param.mean,
                sigma=self.param.sigma,
                USL=self.param.USL,
                LSL=self.param.LSL,
                k1=k1_proposed,
                k2=k2_proposed,
            )
            ax.plot(k1_proposed, k2_proposed, cpk_proposed, c="r", ls="--", alpha=1)

        ax.view_init(azim=10)
        ax.set_xlabel("$k_1$", color="b")
        ax.set_ylabel("$k_2$", color="b")
        ax.set_zlabel("$dynamic$ $C_{pk}$", color="r")
        ax.set_xlim(0, k1_max + step)
        ax.set_ylim(1, k2_max + step)

        plt.savefig(save_path, dpi=self.plot_conf.dpi, bbox_inches="tight")
        plt.close()

    def plot_power_contourf(
        self,
        save_path="./power_contourf.png",
        n=5,
        k1_max=2,
        k2_max=3,
        step=0.01,
    ):

        # creating k1 and k2 array
        k1_right = np.array(
            [float(k1) for k1 in self.plot_conf.k2_df.loc[n - 2, :].index]
        )
        k2_right = self.plot_conf.k2_df.loc[n - 2, :].values
        k1_left = -1 * np.flip(k1_right)
        k2_left = np.flip(k2_right)
        k1 = np.append(k1_left, k1_right)
        k2 = np.append(k2_left, k2_right)

        x = np.arange(-k1_max, k1_max + step, step)
        y = np.arange(step, k2_max + step, step)
        xx, yy = np.meshgrid(x, y)
        z = self.chart.power(xx, yy, n)
        plt.contourf(xx, yy, z)
        plt.plot(k1, k2, c="r", ls="--")
        plt.xlabel("$k_1$")
        plt.ylabel("$k_2$")
        plt.xlim(-2, 2)
        plt.ylim(0, 3)
        plt.savefig(save_path, dpi=self.plot_conf.dpi, bbox_inches="tight")
        plt.close()
