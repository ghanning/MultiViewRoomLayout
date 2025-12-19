from typing import List, Union

import numpy as np

BOLD = "\033[1m"
ENDC = "\033[0m"
MAGENTA = "\033[35m"


class Metric:
    """! Error metric class."""

    def __init__(self, name: str = "", unit: str = "") -> None:
        """! Class initializer.

        @param name Name of metric.
        @param unit Unit.
        """
        self.name = name
        self.unit = unit
        self.values = list()

    def add(self, value: Union[float, List[float]]) -> None:
        """! Add a value.

        @param value Value to add.
        """
        if isinstance(value, list):
            self.values.extend(value)
        else:
            self.values.append(value)

    def min(self) -> float:
        """! Get minimum value.

        @return The minimum value.
        """
        return np.min(self.values)

    def max(self) -> float:
        """! Get maximum value.

        @return The maximum value.
        """
        return np.max(self.values)

    def mean(self) -> float:
        """! Get mean value.

        @return The mean value.
        """
        return np.mean(self.values)

    def median(self) -> float:
        """! Get median value.

        @return The median value.
        """
        return np.median(self.values)

    def auc(self, threshold: float) -> float:
        """! Compute area under the cumulative error curve (AUC).

        @param Threshold Error threshold.
        @return The AUC.
        """
        values, counts = np.unique(self.values, return_counts=True)
        recall = np.cumsum(counts) / np.sum(counts)
        values = np.r_[0.0, values]
        recall = np.r_[0.0, recall]
        idx = np.searchsorted(values, threshold)
        y = np.r_[recall[:idx], recall[idx - 1]]
        x = np.r_[values[:idx], threshold]
        return float(np.trapezoid(y, x)) / threshold

    def summary(self, auc_thr: list = []) -> str:
        """! Get summary.

        @param auc_thr AUC thresholds.
        @return The summary.
        """
        s = f"{MAGENTA}{self.name}{ENDC}\n"
        s += f"{BOLD}Count{ENDC} : {len(self.values)}\n"
        s += f"{BOLD}Min{ENDC}   : {self.min():.3f} {self.unit}\n"
        s += f"{BOLD}Max{ENDC}   : {self.max():.3f} {self.unit}\n"
        s += f"{BOLD}Mean{ENDC}  : {self.mean():.3f} {self.unit}\n"
        s += f"{BOLD}Median{ENDC}: {self.median():.3f} {self.unit}\n"
        for thr in auc_thr:
            s += f"{BOLD}AUC@{thr:2d}{ENDC}: {self.auc(thr):.3f}\n"
        return s

    def print(self, auc_thr: list = []) -> None:
        """! Print summary.

        @param auc_thr AUC thresholds.
        """
        if self.values:
            print(self.summary(auc_thr))
