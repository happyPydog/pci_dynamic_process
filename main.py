import json
from warnings import catch_warnings
import numpy as np
import pandas as pd
from dypro.dynamic import NormalMeanVarChart, NormalMeanSChart, NormalMeanRChart
from dypro.config import Parameters, AdjConf, PlotConf
from dypro.create_csv import (
    create_proposed_cpk,
    created_proposed_yeild,
    create_previous_cpk,
)
from dypro.dynamic.optimize import BrenthOptimizer
from dypro.plot import PlotGraph
from dypro._decorator import RunTime

CHART_LIST = [NormalMeanVarChart(), NormalMeanSChart(), NormalMeanRChart()]
CHART_NAME = ["v", "s", "r"]
FIGNAME = [
    "$S^2$ control chart",
    "$S$ control chart",
    "$R$ control chart",
]
K2_DIR = ["csv/v_k2.csv", "csv/s_k2.csv", "csv/r_k2.csv"]
SUBGROUP_SIZE = [5, 10, 15, 20]
N = 5


@RunTime()
def main():
    # load config
    with open("conf.json") as f:
        conf = json.load(f)

    # create instance object for config
    param = Parameters(**conf["parameters"])
    adj_conf = AdjConf(
        n=np.arange(2, conf["n_max"] + 1),
        k1=np.arange(0, conf["k1_max"] + conf["k1_num"], conf["k1_num"]),
    )

    for chart, k2_dir, chartname, figname in zip(
        CHART_LIST, K2_DIR, CHART_NAME, FIGNAME
    ):

        # optimizer
        optimizer = BrenthOptimizer(chart, power=conf["power"])

        # read k2 table
        k2_df = pd.read_csv(k2_dir)

        # setting plot_conf
        plot_conf = PlotConf(k2_df=k2_df, figsize=conf["figsize"], dpi=conf["dpi"])

        ################
        # create table #
        ################
        # proposed cpk table
        proposed_df = create_proposed_cpk(chart=chart, k2_df=k2_df, param=param)
        proposed_df.to_csv(f"csv/proposed_cpk_{chartname}.csv", index=False)

        # previous
        pre_df = create_previous_cpk(
            chart=chart, optimizer=optimizer, k2_df=k2_df, param=param
        )
        pre_df.to_csv(f"csv/previous_cpk_{chartname}.csv", index=False)

        # yeild table
        yeild_df = created_proposed_yeild(
            proposed_csv=proposed_df, optimizer=optimizer, k2_df=k2_df, param=param
        )
        yeild_df.to_csv(f"csv/yeild_{chartname}.csv", index=False)

        ###########################################
        # plot 2D result with specific parameters #
        ###########################################
        bothe_k1 = np.array(optimizer.get_mean_adjustment(adj_conf.n))
        pearn_k2 = np.array(
            [optimizer.get_var_adjustment(n) for n in adj_conf.n]
        ).flatten()

        plotter = PlotGraph(
            chart=chart,
            proposed_df=proposed_df,
            param=param,
            adj_conf=adj_conf,
            plot_conf=plot_conf,
            bothe_k1=bothe_k1,
            pearn_k2=pearn_k2,
            figname=figname,
        )
        plotter.cpk(save_path=f"fig/cpk_comparison_{chartname}.png")
        plotter.ncppm(save_path=f"fig/cpk_comparison_{chartname}.png")
        plotter.k1_power(
            subgroup_size=SUBGROUP_SIZE,
            save_path=f"fig/k1_power_{chartname}.png",
            k1_max=3,
        )
        plotter.k2_power(
            subgroup_size=SUBGROUP_SIZE,
            save_path=f"fig/k2_power_{chartname}.png",
            k2_max=3,
        )
        plotter.k1_k2_power(
            n=5, save_path=f"fig/k1_k2_power_{chartname}.png", k1_max=3, k2_max=3
        )

        ################
        # plot surface #
        ################

        # scaping r chart for reducing run time
        if chartname != "r":
            plotter.plot_power_surface(
                save_path=f"fig/power_surface_N={N}_{chartname}.png", n=N
            )
            plotter.plot_power_surface(
                save_path=f"fig/power_surface_add_power_line_N={N}_{chartname}.png",
                n=N,
                alpha=0.5,
                add_power_line=True,
            )
            plotter.plot_power_contourf(
                save_path=f"fig/power_contourf_N={N}_{chartname}.png", n=N
            )


if __name__ == "__main__":
    main()
