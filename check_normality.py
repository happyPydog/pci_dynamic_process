import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use(["science", "ieee"])


def main():
    df = pd.read_csv("csv/example_6-1.csv")
    data = df.values.flatten()

    # create Q-Q plot with 45-degree line added to plot
    fig = sm.qqplot(data, line="45", fit=True)
    plt.savefig("QQ-plot.png", dpi=1200)
    plt.show()


if __name__ == "__main__":
    main()
