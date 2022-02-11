import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dypro.pci import functional as F

CSV_DIR = ["csv/proposed_r.csv", "csv/proposed_s.csv", "csv/proposed_v.csv"]
SAMPLE_SIZE = np.arange(2, 31)
MEAN = 12.086
SIGMA = 0.327
USL = 14.0
LSL = 10.0


def main():

    r_csv = pd.read_csv("csv/proposed_cpk_r.csv")
    s_csv = pd.read_csv("csv/proposed_cpk_s.csv")
    v_csv = pd.read_csv("csv/proposed_cpk_v.csv")

    with plt.style.context(["science", "ieee"]):
        fig, ax = plt.subplots()
        plt_param = dict(xlabel="$n$", ylabel="$Dynamic$ $C_{pk}$")
        ax.plot(
            SAMPLE_SIZE,
            F.dynamic_cpk(MEAN, SIGMA, USL, LSL, r_csv["k1 min"], r_csv["k2 min"]),
            label="$R$ chart",
        )
        ax.plot(
            SAMPLE_SIZE,
            F.dynamic_cpk(MEAN, SIGMA, USL, LSL, s_csv["k1 min"], s_csv["k2 min"]),
            label="$S$ chart",
        )
        ax.plot(
            SAMPLE_SIZE,
            F.dynamic_cpk(MEAN, SIGMA, USL, LSL, v_csv["k1 min"], v_csv["k2 min"]),
            label="$S^2$ chart",
        )
    ax.legend(loc="lower right")
    ax.autoscale(tight=True)
    ax.set(**plt_param)
    fig.savefig("chats_comparision.png", dpi=1000, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
