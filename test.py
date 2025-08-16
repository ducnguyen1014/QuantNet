import numpy as np
import pandas as pd


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


# Example DataFrame
df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [10, 12, 23, 23, 16]})

# Calculate MAD for column A
print("MAD of A:", mad(df["A"]))

# Calculate MAD for column B
print("MAD of B:", mad(df["B"]))
