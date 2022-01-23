import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dypro.pci import functional as F

CSV_DIR = ["csv/proposed_r.csv", "csv/proposed_s.csv", "csv/proposed_v.csv"]
SAMPLE_SIZE = np.arange(2, 31)
MEAN = 1.506
SIGMA = 0.1398
USL = 2.0
LSL = 1.0


def main():

    r_csv = pd.read_csv("example_result/proposed_cpk_r.csv")
    s_csv = pd.read_csv("example_result/proposed_cpk_s.csv")
    v_csv = pd.read_csv("example_result/proposed_cpk_v.csv")

    with plt.style.context(["science", "ieee"]):
        fig, ax = plt.subplots()
        plt_param = dict(xlabel="$n$", ylabel="$Dynamic$ $C_{pk}$")
        ax.plot(
            SAMPLE_SIZE,
            F.dynamic_cpk(MEAN, SIGMA, USL, LSL, r_csv["k1 min"], r_csv["k2 min"]),
            label="$R$ control chart",
        )
        ax.plot(
            SAMPLE_SIZE,
            F.dynamic_cpk(MEAN, SIGMA, USL, LSL, s_csv["k1 min"], s_csv["k2 min"]),
            label="$S$ control chart",
        )
        ax.plot(
            SAMPLE_SIZE,
            F.dynamic_cpk(MEAN, SIGMA, USL, LSL, v_csv["k1 min"], v_csv["k2 min"]),
            label="$S^2$ control chart",
        )
    ax.legend(loc="lower right")
    ax.autoscale(tight=True)
    ax.set(**plt_param)
    fig.savefig("chats_comparision_example_3-1.png", dpi=1000, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
