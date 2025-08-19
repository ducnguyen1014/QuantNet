import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from empyrical import (
    annual_return,
    annual_volatility,
    downside_risk,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    tail_ratio,
    value_at_risk,
)
from scipy.stats import kurtosis, skew


def calmar_ratio(x):
    return annual_return(x).values / -max_drawdown(x)


def compute_performance_metrics(df_returns):
    """

    :param df_returns:
    :return:
    """

    # metrics to compute
    pf_metrics = [
        sharpe_ratio,
        calmar_ratio,
        max_drawdown,
        annual_return,
        annual_volatility,
        sortino_ratio,
        downside_risk,
        value_at_risk,
        tail_ratio,
        skew,
        kurtosis,
    ]
    pf_metrics_labels = [
        "SR",
        "CR",
        "MDD",
        "ANN_RET",
        "ANN_VOL",
        "SortR",
        "DownRisk",
        "VaR",
        "TailR",
        "Skew",
        "Kurt",
    ]

    # compute performance metric
    df_metrics = pd.DataFrame(
        index=range(df_returns.shape[1]), columns=pf_metrics_labels
    )
    for pf, pf_label in zip(pf_metrics, pf_metrics_labels):
        df_metrics[pf_label] = np.array(pf(df_returns))
    df_metrics.index = df_returns.columns

    return df_metrics


def get_data(data_config, problem_config, model_config):
    """

    :return:
    """
    Xtrain_tasks, Xval_tasks, Xtest_tasks = {}, {}, {}
    for region in data_config["region"]:
        # pre-allocation
        region_task_paths = [t + "_all_assets_data.pkl.gz" for t in data_config[region]]
        Xtrain_tasks[region], Xval_tasks[region], Xtest_tasks[region] = {}, {}, {}

        for tk_path, task in zip(region_task_paths, data_config[region]):
            # get data
            df = pd.read_pickle(data_config["data_path"] + tk_path)
            df_train = df.iloc[
                : -(problem_config["val_period"] + problem_config["holdout_period"])
            ]
            if problem_config["val_period"] != 0:
                df_val = df.iloc[
                    -(
                        problem_config["val_period"] + problem_config["holdout_period"]
                    ) : -problem_config["holdout_period"]
                ]
            else:
                df_val = df.iloc[
                    : -(problem_config["val_period"] + problem_config["holdout_period"])
                ]
            df_test = df.iloc[-problem_config["holdout_period"] :]

            # transform in tensor
            Xtrain_tasks[region][task] = torch.from_numpy(df_train.values).to(
                model_config["device"]
            )
            Xval_tasks[region][task] = torch.from_numpy(df_val.values).to(
                model_config["device"]
            )
            Xtest_tasks[region][task] = torch.from_numpy(df_test.values).to(
                model_config["device"]
            )
            print(region, task, Xtrain_tasks[region][task].size())

    return Xtrain_tasks, Xval_tasks, Xtest_tasks


def compute_trading_costs(signal):
    slip = 0.0005 * 0.00
    bp = 0.0020 * 0.00
    tc = torch.abs(signal[:, 1:, :] - signal[:, :-1, :]) * (bp + slip)
    tc = torch.cat([torch.zeros(signal.size(0), 1, signal.size(2)).double(), tc], dim=1)
    return tc


def mad(data) -> float:
    """
    Calculate the Mean Absolute Deviation (MAD).

    Parameters
    ----------
    data : list, numpy.ndarray, or pandas.Series
        Input data.

    Returns
    -------
    float
        Mean absolute deviation.
    """
    arr = np.asarray(data)  # convert to numpy array
    mean_val = np.mean(arr)
    mad = np.mean(np.abs(arr - mean_val))
    return mad


def draw_table(df: pd.DataFrame, upper_prefix: str, lower_prefix: str) -> plt.Figure:
    # Identify strategies dynamically (every M_* has a matching SE_*)
    strategies = sorted(
        {c.split("_", 1)[1] for c in df.columns if c.startswith(f"{upper_prefix}_")}
    )

    # Build rows
    rows = []
    for metric, r in df.iterrows():
        row = [metric]
        for strat in strategies:
            mean_val = r[f"{upper_prefix}_{strat}"]
            se_val = r[f"{lower_prefix}_{strat}"]
            row.append(f"{mean_val:.6f}\n({se_val:.5f})")
        rows.append(row)

    # Column headers
    columns = ["Metric"] + strategies

    # Create figure
    fig, ax = plt.subplots(figsize=(18, 3))
    ax.axis("off")

    # Draw table
    table = ax.table(cellText=rows, colLabels=columns, cellLoc="center", loc="center")

    # Style
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.3, 1.3)

    # Adjust row heights for multi-line text
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#d0d6e0")
            cell.set_text_props(weight="bold")
            cell.set_height(0.25)
        else:
            cell.set_height(0.35)

    plt.show()


def main():
    # Store data with row index
    data = {
        "M_Buy": [0.000020, 0.287515, 0.000040],
        "SE_Buy": [0.13433, 0.10145, 0.33583],
        "M_Risk": [0.000193, 0.001536, 0.095516],
        "SE_Risk": [0.00270, 0.00537, 0.29444],
    }

    index = ["Ann Ret", "Ann Vol", "CR"]

    df = pd.DataFrame(data, index=index)

    print(df)

    draw_table(df=df, upper_prefix="M", lower_prefix="SE")


if __name__ == "__main__":
    main()
