import os
import numpy as np
import matplotlib.pyplot as plt


COLORS = ["orange", "forestgreen", 'blue']


def histogram_overlayer(inv_distortion_values_dict, save_dir):
    # Create save_dir.
    os.makedirs(save_dir, exist_ok=True)

    # Create the plot.
    plt.close("all")
    plt.figure(figsize=(10, 5))
    # ____ Plot the histograms. ____
    for i, (k, v) in enumerate(inv_distortion_values_dict.items()):
        plt.hist(x=v, bins=50, color=COLORS[i], density=True, label=k, alpha=0.3)

    # Add the min, max and mean.
    for i, (k, v) in enumerate(inv_distortion_values_dict.items()):
        v_np = np.array(v)
        plt.axvline(v_np.mean(), color=COLORS[i], linestyle='-', linewidth=1, label=f"{k}/mean")
        plt.axvline(v_np.min(), color=COLORS[i], linestyle='-.', linewidth=1, label=f"{k}/min")

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.xlim(-0.05, 1.05)
    plt.yscale('log')
    plt.savefig(os.path.join(save_dir, "ovelaid_histogram.png"))


