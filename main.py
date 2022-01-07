import json
import pandas as pd
from dypro.dynamic import NormalMeanVarChart, NormalMeanSChart, NormalMeanRChart
from dypro.config import Parameters
from dypro.create_csv import create_proposed_cpk

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
        k2_df = pd.read_csv(k2_dir)

        # create table
        proposed_df = create_proposed_cpk(chart=chart, df=k2_df, param=param)
        proposed_df.to_csv(f"csv/proposed_cpk_{name}.csv", index=False)


if __name__ == "__main__":
    main()
