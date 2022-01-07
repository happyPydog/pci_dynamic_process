import json
import pandas as pd
from dypro.dynamic import NormalMeanVarChart, NormalMeanSChart, NormalMeanRChart
from dypro.config import Parameters
from dypro.create_csv import (
    create_proposed_cpk,
    created_proposed_yeild,
    create_previous_cpk,
)
from dypro.dynamic.optimize import BrenthOptimizer

CHART_NAME = ["v", "s", "r"]
K2_DIR = ["csv/v_k2.csv", "csv/s_k2.csv", "csv/r_k2.csv"]


def main():

    # load config
    with open("conf.json") as f:
        conf = json.load(f)

    param = Parameters(**conf["parameters"])
    chart_list = [NormalMeanVarChart(), NormalMeanSChart(), NormalMeanRChart()]
    # chart_name_list = conf["plot_conf"]["figname"]

    for chart, k2_dir, name in zip(chart_list, K2_DIR, CHART_NAME):

        # optimizer
        optimizer = BrenthOptimizer(chart, power=conf["power"])

        # read k2 table
        k2_df = pd.read_csv(k2_dir)

        ################
        # create table #
        ################
        # proposed cpk table
        proposed_df = create_proposed_cpk(chart=chart, df=k2_df, param=param)
        proposed_df.to_csv(f"csv/proposed_cpk_{name}.csv", index=False)

        # previous
        pre_df = create_previous_cpk(
            chart=chart, optimizer=optimizer, k2_df=k2_df, param=param
        )
        pre_df.to_csv(f"csv/previous_cpk_{name}.csv", index=False)

        # yeild table
        yeild_df = created_proposed_yeild(
            proposed_csv=proposed_df, optimizer=optimizer, k2_df=k2_df, param=param
        )
        yeild_df.to_csv(f"csv/yeild_{name}.csv", index=False)


if __name__ == "__main__":
    main()
