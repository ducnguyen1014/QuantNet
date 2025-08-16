import pandas as pd
import matplotlib.pyplot as plt

# Load your CSV
df = pd.read_csv("chart_2.csv")

# Rename strategies for clarity
df["transferstrat"] = df["transferstrat"].replace({
    "1000_global_lstm_lstm": "QuantNet",
    "1000_no_transfer_linear": "No Transfer"
})

# Pivot to wide format: one column for each strategy
pivot_df = df.pivot(index="exchange", columns="transferstrat", values="SR")

# Plot grouped bar chart
ax = pivot_df.plot(kind="bar", figsize=(14,6), width=0.8)

plt.ylabel("Sharpe ratio")
plt.xlabel("Region_Market")
plt.title("Sharpe ratio comparison: QuantNet vs No Transfer")
plt.axhline(0, color="gray", linewidth=0.8)

plt.xticks(rotation=75, ha="right")
plt.legend(title="")
plt.tight_layout()
plt.show()
