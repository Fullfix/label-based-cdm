import torch
import numpy as np
import logging
import scipy
import pandas as pd


def disable_pl_logger():
    """Disable pytorch lightning logger"""

    log = logging.getLogger("pytorch_lightning")
    log.propagate = False
    log.setLevel(logging.ERROR)


def enable_pl_logger():
    """Enable pytorch lightning logger"""

    log = logging.getLogger("pytorch_lightning")
    log.propagate = True
    log.setLevel(logging.INFO)


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model"""

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_confidence_interval(samples: list, confidence=0.95):
    """Compute the confidence interval for a list of samples"""

    mean = np.mean(samples)
    n = len(samples)
    stderr = scipy.stats.sem(samples)
    t_value = scipy.stats.t.ppf((1 + confidence) / 2, n - 1)
    offset = t_value * stderr
    return mean, offset


def generate_latex_corr_table(df: pd.DataFrame, formatter=None) -> str:
    """Generate correlation table for copypasting to latex"""

    latex_code = r"\begin{tabular}{|c|" + "c|" * len(df.columns) + r"}\hline" + "\n"

    latex_code += "       & " + " & ".join(df.columns) + r" \\" + r" \hline" + "\n"

    def formatter_fn(x, i, j):
        if formatter is not None:
            return formatter(x, i, j)
        return x

    for i, row in enumerate(df.index):
        row_content = f"    {row}    & "  # Add row name
        row_content += " & ".join(
            [formatter_fn("{:.2f}".format(df.iloc[i, j]), i, j) if j >= i else "" for j in range(len(df.columns))]
        )
        latex_code += row_content + r" \\" + r" \hline" + "\n"

    latex_code += r"\end{tabular}"
    return latex_code


def generate_latex_table(df: pd.DataFrame, formatter=None) -> str:
    """Generate table for copypasting to latex"""

    latex_code = r"\begin{tabular}{|c|" + "c|" * len(df.columns) + r"}\hline" + "\n"

    latex_code += "       & " + " & ".join(df.columns) + r" \\" + r" \hline" + "\n"

    def formatter_fn(x, i, j):
        if formatter is not None:
            return formatter(x, i, j)
        return x

    for i, row in enumerate(df.index):
        row_content = f"    {row}    & "  # Add row name
        row_content += " & ".join(
            [formatter_fn("{:.2f}".format(df.iloc[i, j]), i, j) for j in range(len(df.columns))]
        )
        latex_code += row_content + r" \\" + r" \hline" + "\n"

    latex_code += r"\end{tabular}"
    return latex_code