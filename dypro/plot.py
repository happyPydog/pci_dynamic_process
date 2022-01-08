from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .dynamic import BaseChart
from .pci import functional as F
from .config import Parameters, AdjConf, PlotConf


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
            plt_param = dict(
                xlabel="$n$",
                ylabel="$Dynamic$ $C_{pk}$",
                # title="Different Measurement for Dynamic $C_{pk}$ with " + self.figname,
            )
            ax.plot(
                self.adj_conf.n,
                F.dynamic_cpk(mean, sigma, USL, LSL, k1, k2),
                label="Proposed Method",
            )
            ax.plot(
                self.adj_conf.n,
                F.dynamic_cpk(mean, sigma, USL, LSL, self.bothe_k1, 1),
                label="Mean Shift",
            )
            ax.plot(
                self.adj_conf.n,
                F.dynamic_cpk(mean, sigma, USL, LSL, 0, self.pearn_k2),
                label="Variance Change",
            )
            ax.plot(
                self.adj_conf.n,
                F.dynamic_cpk(mean, sigma, USL, LSL, self.bothe_k1, self.pearn_k2),
                label="Tai",
            )

            ax.legend(loc="lower right")
            ax.autoscale(tight=True)
            ax.set(**plt_param)
            fig.savefig(save_path, dpi=self.plot_conf.dpi, bbox_inches="tight")

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
            plt_param = dict(
                xlabel="$n$",
                ylabel="$Dynamic$ $C_{pk}$",
                # title="Different Measurement for Dynamic $C_{pk}$ with " + self.figname,
            )
            ax.plot(
                self.adj_conf.n,
                F.ncppm(F.dynamic_cpk(mean, sigma, USL, LSL, k1, k2)),
                label="Proposed Method",
            )
            ax.plot(
                self.adj_conf.n,
                F.ncppm(F.dynamic_cpk(mean, sigma, USL, LSL, self.bothe_k1, 1)),
                label="Mean Shift",
            )
            ax.plot(
                self.adj_conf.n,
                F.ncppm(F.dynamic_cpk(mean, sigma, USL, LSL, 0, self.pearn_k2)),
                label="Variance Change",
            )
            ax.plot(
                self.adj_conf.n,
                F.ncppm(
                    F.dynamic_cpk(mean, sigma, USL, LSL, self.bothe_k1, self.pearn_k2)
                ),
                label="Tai",
            )

            ax.legend(loc="lower right")
            ax.autoscale(tight=True)
            ax.set(**plt_param)
            fig.savefig(save_path, dpi=self.plot_conf.dpi, bbox_inches="tight")

    def k1_power(self, subgroup_size: list[int], save_path, k1_max=3):
        k1 = np.arange(0, k1_max + 0.01, 0.01)
        plt_param = dict(
            xlabel="$k_1$",
            ylabel="$Power$",
            ylim=[0, 1],
            # title=f"Detection Power for Various Sample sizes with {self.figname}",
        )

        with plt.style.context(["science", "ieee"]):
            fig, ax = plt.subplots()
            for n in subgroup_size:
                ax.plot(k1, self.chart.power(k1, 1, n), label=f"n={n}")
            ax.legend(loc="lower right", title="Sample Size")
            ax.autoscale(tight=True)
            ax.set(**plt_param)
            fig.savefig(save_path, dpi=self.plot_conf.dpi, bbox_inches="tight")

    def k2_power(self, subgroup_size: list[int], save_path, k2_max=3):
        k2 = np.arange(1, k2_max + 0.01, 0.01)
        plt_param = dict(
            xlabel="$k_2$",
            ylabel="$Power$",
            ylim=[0, 1],
            # title=f"Detection Power for Various Sample sizes with {self.figname}",
        )

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
        plt_param = dict(
            xlabel="$k_1$ and $k_2$",
            ylabel="$Power$",
            ylim=[0, 1],
            # title=f"Detection Power for Various Sample sizes with {self.figname}",
        )

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
