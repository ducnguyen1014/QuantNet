import matplotlib.pyplot as plt
import pandas as pd

# Example dataframe with multi-level columns
data = {
    ("Height", "mean"): [170, 165, 180],
    ("Height", "sd"): [5, 6, 4],
    ("Weight", "mean"): [65, 70, 80],
    ("Weight", "sd"): [3, 4, 5],
}
df = pd.DataFrame(data)

# Prepare column labels as two rows
columns = pd.MultiIndex.from_tuples(df.columns)
col_labels = [columns.get_level_values(0), columns.get_level_values(1)]

fig, ax = plt.subplots(figsize=(6, 3))
ax.axis("off")

# Draw the table (data only, no headers)
table = ax.table(
    cellText=df.values,
    rowLabels=[f"Row {i}" for i in df.index],
    colLabels=col_labels[1],  # bottom-level labels
    cellLoc="center",
    loc="center",
)

# Add the top-level labels manually
for j, label in enumerate(col_labels[0]):
    table.add_cell(-2, j, width=table[0, j].get_width(), height=0.2)
    table[-2, j].get_text().set_text(label)
    table[-2, j].set_fontsize(10)
    table[-2, j].set_facecolor("#f0f0f0")

plt.show()
