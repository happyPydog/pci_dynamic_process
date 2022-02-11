from typing import Protocol
import numpy as np
import pandas as pd
from scipy.stats import chi2, chi
from ..pci import functional as F


class VarChange(Protocol):
    def beta(self, k2, n, alpha):
        ...

    def power(self, k2, n, alpha):
        ...


class NormalVarChange:
    """Variance chart dynamic process with variance change under normal distribution."""

    def beta(self, k2, n, alpha):
        chi_square_right = chi2.ppf((1 - alpha / 2), n - 1)
        chi_square_left = chi2.ppf(alpha / 2, n - 1)
        UCL = chi_square_right / (k2 ** 2)
        LCL = chi_square_left / (k2 ** 2)

        return chi2.cdf(UCL, n - 1) - chi2.cdf(LCL, n - 1)

    def power(self, k2, n, alpha):
        return 1 - self.beta(k2, n, alpha)


class NormalSChange:
    """S chart dynamic process with variance change under normal distribution."""

    def beta(self, k2, n, alpha):
        UCL = F.B4(n) * np.sqrt(n - 1) / (k2)
        LCL = F.B3(n) * np.sqrt(n - 1) / (k2)

        return chi.cdf(UCL, n - 1) - chi.cdf(LCL, n - 1)

    def power(self, k2, n, alpha):
        return 1 - self.beta(k2, n, alpha)


class NormalRChange:
    """R chart dynamic process with variance change under normal distribution."""

    def __init__(self, facotors_path: str):
        self.table = pd.read_csv(facotors_path)

    def beta(self, k2, n, alpha):
        if hasattr(self, "table"):
            assert np.all(n <= 30), "foctor_table.csv only contain numbers to 30."
            index = np.where(self.table.n == n)[0]
            UCL = self.table.R_right[index] / k2
            LCL = self.table.R_left[index] / k2

        else:
            UCL = F.R_right(n, x0=5, alpha=alpha) / k2
            LCL = F.R_left(n, x0=2, alpha=alpha) / k2

        return F.w_cdf(UCL, n) - F.w_cdf(LCL, n)

    def power(self, k2, n, alpha):
        return 1 - self.beta(k2, n, alpha)
