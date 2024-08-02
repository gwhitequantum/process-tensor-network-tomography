import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[1]

# Construct the path to the data directory
data_dir = project_root / "data"
est_dir = data_dir / "DD_opt" / "DD_predictions"

import seaborn as sns

sns.set_palette("deep")
sns.set_style("whitegrid")

file_path = "DD_job_estimates.txt"
file_names = []
# Open the file in read mode
with open(file_path, "r") as file:
    # Iterate over each line in the file
    for line in file:
        # Process the string on each line (e.g., print it)
        file_names.append(line.strip())

data_frames = []
data_names = []
for file in file_names:
    tmp_df = pickle.load(open(str(est_dir) + "/" + file, "rb"))
    data_frames.append(tmp_df)
    data_names.append(file)

    # new_df = 0.5 * tmp_df.sort_values("identity", ascending=False)[50:]
    # new_df.reset_index(drop=True, inplace=True)

    # with open(str(est_dir) + "/" + file, "wb") as handle:
    #     pickle.dump(new_df, handle)
    # new_df =

# Regular expression pattern
pattern = r"([^_]+)_(\d+)_"

num_rows = int(np.ceil(len(data_frames) / 4))


def plot_identity(data_frame, ax, minval=0.05):
    data_from = 0
    data_size = 50

    sorted_data_frame = data_frame.sort_values("identity", ascending=False)

    ax.scatter(
        range(data_size),
        sorted_data_frame["identity"][data_from:],
        alpha=0.75,
        label="Idle",
        s=7,
        color="C2",
    )
    # ax.errorbar(range(100), sorted_data_frame['identity'], yerr = y_err, elinewidth = 0.1, ecolor='black', capsize=3, capthick=0.1)
    ax.plot(
        range(data_size),
        [np.median(sorted_data_frame["identity"]) for i in range(data_size)],
        "--",
        color="C2",
    )

    X = np.median(sorted_data_frame["identity"])
    ax.text(
        data_size - 0.5,
        X,
        f"{X:.2f}",
        color="C2",
        verticalalignment="center",
        horizontalalignment="left",
    )

    ax.fill_between(
        range(data_size),
        [minval for i in range(data_size)],
        list(sorted_data_frame["identity"][data_from:]),
        alpha=0.075,
        color="green",
    )

    # ax.set_xlabel('Sequence Number')
    # ax.set_ylabel('Distance from Ideal')

    return None


def plot_DD(data_frame, ax, minval=0.05):

    # data_from = 50
    # data_size = 100 - data_from
    data_from = 0
    data_size = 50
    sorted_data_frame = data_frame.sort_values("identity", ascending=False)

    ax.scatter(
        range(data_size),
        sorted_data_frame["XY"][data_from:],
        alpha=0.75,
        label="XY4",
        s=7,
    )
    ax.scatter(
        range(data_size),
        sorted_data_frame["optimised"][data_from:],
        alpha=0.75,
        label="Opt",
        s=7,
    )

    ax.plot(
        range(data_size),
        [np.median(sorted_data_frame["XY"]) for i in range(data_size)],
        "--",
    )
    X = np.median(sorted_data_frame["XY"])
    ax.text(
        data_size - 0.5,
        X,
        f"{X:.2f}",
        color="C0",
        verticalalignment="center",
        horizontalalignment="left",
    )

    ax.plot(
        range(data_size),
        [np.median(sorted_data_frame["optimised"]) for i in range(data_size)],
        "--",
    )
    X = np.median(sorted_data_frame["optimised"])
    ax.text(
        data_size - 0.5,
        X,
        f"{X:.2f}",
        color="C1",
        verticalalignment="center",
        horizontalalignment="left",
    )

    ax.fill_between(
        range(data_size),
        [minval for i in range(data_size)],
        list(sorted_data_frame["XY"][data_from:]),
        alpha=0.075,
        color="blue",
    )
    ax.fill_between(
        range(data_size),
        [minval for i in range(data_size)],
        list(sorted_data_frame["optimised"][data_from:]),
        alpha=0.075,
        color="orange",
    )

    ax.set_xlabel("Sequence Number")
    # ax.set_ylabel('Trace distance from Ideal')

    return None


for i, data_frame in enumerate(data_frames):
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(4, 3))

    match = re.match(pattern, data_names[i])
    device_name = match.group(1)
    number_of_steps = int(match.group(2))

    axs[0].set_title(f"ibm_{device_name}, {number_of_steps-1} pulses")
    # ax.semilogy()
    # Plot data using your function
    minval = np.min(data_frame["optimised"]) - 0.025

    plot_identity(data_frame, axs[0], minval)
    plot_DD(data_frame, axs[1], minval)

    axs[0].set_xlim(None, 55)

    fig.text(0.005, 0.55, "Trace distance from ideal", va="center", rotation="vertical")

    plt.subplots_adjust(hspace=0.05)

    plt.tight_layout()
    name = data_names[i][:-7] + ".pdf"
    plt.savefig(name, bbox_inches="tight")
    # Show the plot
    # plt.show()
