"""
Check all folders in generalization/Llama-2-7b/few_shot_prompting/ 
If there is only a json file, then also save the plot in the same folder.
"""

import os
from pathlib import Path
from display_utils import plot_results
import json
import matplotlib.pyplot as plt
import torch as th

root = Path(__file__).resolve().parent / "results" / "Llama-2-7b" / "few_shot_prompting"
for folder in os.listdir(root):
    folder_path = root / folder
    if not folder_path.is_dir():
        continue
    # if len(os.listdir(folder_path)) == 1:
    if True:
        # get the json file
        json_file = list(folder_path.glob("*.json"))[0]
        # load the json file
        with open(json_file, "r") as f:
            data = json.load(f)
        # get the plot
        keys = list(data.keys())
        probs = data.pop(keys[0])
        fig, ax = plt.subplots()
        plot_results(ax, th.tensor(probs), {k: th.tensor(v) for k,v in data.items()}, keys[0])
        ax.legend()
        ax.set_title(f"Llama-2-7b - {folder_path.name}")
        # save the plot in the same folder with the same name as the json file
        plt.savefig(folder_path / (str(json_file.stem) + ".png"))
        plt.close()
