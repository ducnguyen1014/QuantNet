import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# ---- Example Data ----
np.random.seed(42)
n = 300
df = pd.DataFrame(
    {
        "Sharpe ratio": np.concatenate(
            [
                np.random.normal(0.5, 0.7, n),
                np.random.normal(1.0, 0.6, n),
                np.random.normal(0.2, 0.8, n),
                np.random.normal(0.8, 0.7, n),
                np.random.normal(0.3, 0.9, n),
                np.random.normal(1.2, 0.6, n),
                np.random.normal(0.1, 0.7, n),
                np.random.normal(1.0, 0.5, n),
            ]
        ),
        "Method": ["No Transfer"] * n
        + ["QuantNet"] * n
        + ["No Transfer"] * n
        + ["QuantNet"] * n
        + ["No Transfer"] * n
        + ["QuantNet"] * n
        + ["No Transfer"] * n
        + ["QuantNet"] * n,
        "exchange": (
            ["Europe_UKX"] * 2 * n
            + ["Americas_SPX"] * 2 * n
            + ["MEA_SASEIDX"] * 2 * n
            + ["Asia and Pacific_KOSPI"] * 2 * n
        ),
    }
)

# ---- Plot ----
sns.set(style="darkgrid", rc={"axes.facecolor": "#d6e0f0"})  # background like yours

g = sns.FacetGrid(
    df,
    col="exchange",
    col_wrap=2,
    hue="Method",
    sharex=True,
    sharey=True,
    height=3.5,
    aspect=1.5,
)

# Histogram with KDE
g.map(
    sns.histplot, "Sharpe ratio", stat="density", kde=False, alpha=0.3, multiple="layer"
)
g.map(sns.kdeplot, "Sharpe ratio", lw=2)

# Adjust legend
g.add_legend(title="Method")

# Titles and labels
for ax in g.axes.flat:
    ax.set_xlabel("Sharpe ratio")
    ax.set_ylabel("")

plt.tight_layout()
plt.show()
