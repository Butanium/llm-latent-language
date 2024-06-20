import json
import gzip
import _pickle as pickle
import logging
import yaml
import json
from logging import Logger
import re
from datetime import datetime
import os
import pytz
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd
import torch as th
from transformers import StoppingCriteria
from matplotlib import markers, font_manager
from pathlib import Path
from transformer_lens.loading_from_pretrained import OFFICIAL_MODEL_NAMES, MODEL_ALIASES
from transformer_lens import HookedTransformerKeyValueCache as KeyValueCache

from nnsight import LanguageModel
from nnsight.models.UnifiedTransformer import UnifiedTransformer
from contextlib import nullcontext
from IPython.display import display


PATH = Path(os.path.dirname(os.path.realpath(__file__)))

markers_list = [None] + list(markers.MarkerStyle.markers.keys())
simsun_path = PATH / "data/SimSun.ttf"
font_manager.fontManager.addfont(str(simsun_path))
simsun = font_manager.FontProperties(fname=str(simsun_path)).get_name()

plt.rcParams.update({"font.size": 16})
plt_params = dict(linewidth= 2.7, alpha= 0.8, linestyle="-", marker="o")


def plot_ci_plus_heatmap(
    data,
    heat,
    labels,
    color="blue",
    linestyle="-",
    tik_step=10,
    method="gaussian",
    init=True,
    do_colorbar=False,
    shift=0.5,
    nums=[0.99, 0.18, 0.025, 0.6],
    labelpad=10,
    plt_params=plt_params,
):

    fig, (ax, ax2) = plt.subplots(
        nrows=2, sharex=True, gridspec_kw={"height_ratios": [1, 10]}, figsize=(5, 3)
    )
    if do_colorbar:
        fig.subplots_adjust(right=0.8)
    plot_ci(
        ax2,
        data,
        labels,
        color=color,
        linestyle=linestyle,
        tik_step=tik_step,
        method=method,
        init=init,
        plt_params=plt_params,
    )

    y = heat.mean(dim=0)
    x = np.arange(y.shape[0]) + 1

    extent = [
        x[0] - (x[1] - x[0]) / 2.0 - shift,
        x[-1] + (x[1] - x[0]) / 2.0 + shift,
        0,
        1,
    ]
    img = ax.imshow(
        y[np.newaxis, :], cmap="plasma", aspect="auto", extent=extent, vmin=0, vmax=14
    )
    ax.set_yticks([])
    # ax.set_xlim(extent[0], extent[1])
    if do_colorbar:
        cbar_ax = fig.add_axes(nums)  # Adjust these values as needed
        cbar = plt.colorbar(img, cax=cbar_ax)
        cbar.set_label(
            "entropy", rotation=90, labelpad=labelpad
        )  # Adjust label and properties as needed
    plt.tight_layout()
    return fig, ax, ax2


def plot_ci(
    ax,
    data,
    label,
    color="blue",
    linestyle="-",
    tik_step=10,
    init=True,
    plt_params=plt_params,
    marker="o",
):
    if init:
        upper = max(round(data.shape[1] / 10) * 10 + 1, data.shape[1] + 1)
        ax.set_xticks(np.arange(0, upper, tik_step))
        for i in range(0, upper, tik_step):
            ax.axvline(i, color="black", linestyle="--", alpha=0.5, linewidth=1)
    mean = data.mean(dim=0)
    std = data.std(dim=0)
    data_ci = {
        "x": np.arange(data.shape[1]),
        "y": mean,
        "y_upper": mean + (1.96 / (data.shape[0] ** 0.5)) * std,
        "y_lower": mean - (1.96 / (data.shape[0] ** 0.5)) * std,
    }

    df = pd.DataFrame(data_ci)
    # Create the line plot with confidence intervals
    ax.plot(
        df["x"], df["y"], label=label, color=color, **plt_params
    )
    ax.fill_between(df["x"], df["y_lower"], df["y_upper"], color=color, alpha=0.3)
    if init:
        ax.spines[["right", "top"]].set_visible(False)


def plot_k(
    axes, data, label, k=4, color="blue", tik_step=10, plt_params=plt_params, init=True, same_scale=True
):
    if len(axes) < k:
        raise ValueError("Number of axes must be greater or equal to k")

    for i in range(k):
        ax = axes[i]
        if init:
            upper = max(round(data.shape[1] / 10) * 10 + 1, data.shape[1] + 1)
            ax.set_xticks(np.arange(0, upper, tik_step))
            for j in range(0, upper, tik_step):
                ax.axvline(j, color="black", linestyle="--", alpha=0.5, linewidth=1)
        ax.plot(data[i], label=label, color=color, **plt_params)
        if init:
            ax.spines[["right", "top"]].set_visible(False)
        if same_scale and init:
            ax.set_ylim(0, 1)


def yaml_to_dict(yaml_file):
    with open(yaml_file, "r") as file:
        return yaml.safe_load(file)


def save_pickle(file, path):
    with open(path, "wb") as f:
        pickle.dump(file, f)


def load_pickle(path):
    if path.endswith("gz"):
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    with open(path, "rb") as f:
        return pickle.load(f)


def printr(text):
    print(f"[running]: {text}")


def save_json(data: object, json_path: str) -> None:
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def prepare_output_dir(base_dir: str = "./runs/") -> str:
    # create output directory based on current time (using zurich time zone)
    experiment_dir = os.path.join(
        base_dir,
        datetime.now(tz=pytz.timezone("Europe/Zurich")).strftime("%Y-%m-%d_%H-%M-%S"),
    )
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


def get_logger(output_dir) -> Logger:
    os.makedirs(os.path.dirname(LOG_DIR), exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(filename)s - %(levelname)s - %(message)s"
    )

    # Log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Log to file
    file_path = os.path.join(
        LOG_DIR, f'{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.log'
    )
    fh = logging.FileHandler(os.path.join(output_dir, "log.txt"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def get_api_key(fname, provider="azure", key=None):
    print(fname)
    try:
        with open(fname) as f:
            keys = json.load(f)[provider]
            if key is not None:
                api_key = keys[key]
            else:
                api_key = list(keys.values())[0]
    except Exception as e:
        print(f"error: unable to load {provider} api key {key} from file {fname} - {e}")
        return None

    return api_key


def read_json(path_name: str):
    with open(path_name, "r") as f:
        json_file = json.load(f)
    return json_file


def printv(msg, v=0, v_min=0, c=None, debug=False):
    # convenience print function
    if debug:
        c = "yellow" if c is None else c
        v, v_min = 1, 0
        printc("\n\n>>>>>>>>>>>>>>>>>>>>>>START DEBUG\n\n", c="yellow")
    if (v > v_min) or debug:
        if c is not None:
            printc(msg, c=c)
        else:
            print(msg)
    if debug:
        printc("\n\nEND DEBUG<<<<<<<<<<<<<<<<<<<<<<<<\n\n", c="yellow")


def printc(x, c="r"):
    m1 = {
        "r": "red",
        "g": "green",
        "y": "yellow",
        "w": "white",
        "b": "blue",
        "p": "pink",
        "t": "teal",
        "gr": "gray",
    }
    m2 = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "pink": "\033[95m",
        "teal": "\033[96m",
        "white": "\033[97m",
        "gray": "\033[90m",
    }
    reset_color = "\033[0m"
    print(f"{m2.get(m1.get(c, c), c)}{x}{reset_color}")


def extract_dictionary(x):
    if isinstance(x, str):
        regex = r"{.*?}"
        match = re.search(regex, x, re.MULTILINE | re.DOTALL)
        if match:
            try:
                json_str = match.group()
                json_str = json_str.replace("'", '"')
                dict_ = json.loads(json_str)
                return dict_
            except Exception as e:
                print(f"unable to extract dictionary - {e}")
                return None

        else:
            return None
    else:
        return None


class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_tokens):
        """
        Args:
            stop_tokens (int or List[int]): The token(s) to stop generation at.
        """
        if isinstance(stop_tokens, int):
            stop_tokens = [stop_tokens]
        self.stop_tokens = stop_tokens

    def __call__(self, input_ids, _scores, **_kwargs):
        if input_ids[0][-1] in self.stop_tokens:
            return True  # Stop generation
        return False  # Continue generation

    def __len__(self):
        return 1

    def __iter__(self):
        yield self

    def from_string(string, tokenizer):
        """
        Initialize the stop tokens as all the tokens that start or end with the given string.
        """
        stop_tokens = [
            i
            for i in range(tokenizer.vocab_size)
            if tokenizer.decode(i).startswith(string)
            or tokenizer.decode(i).endswith(string)
            or string in tokenizer.decode(i)
        ]
        return StopOnTokens(stop_tokens)


class StopOnSequence(StoppingCriteria):
    def __init__(self, stop_sequence):
        """
        Args:
            stop_sequence (List[int]): The sequence to stop generation at.
        """
        self.stop_sequence = stop_sequence
        self.state = 0

    def __call__(self, input_ids, _scores, **_kwargs):
        if input_ids[0][-1] == self.stop_sequence[self.state]:
            self.state += 1
            if self.state == len(self.stop_sequence):
                return True
        else:
            self.state = 0
        return False

    def __len__(self):
        return 1

    def __iter__(self):
        yield self

    def from_string(string, tokenizer):
        """
        Initialize the stop tokens as all the tokens that start or end with the given string.
        """
        stop_sequence = [tokenizer(string, add_special_tokens=False)]
        return StopOnSequence(stop_sequence)


def add_model_to_transformer_lens(official_name, alias=None):
    """
    Hacky way to add a model to transformer_lens even if it's not in the official list.
    """
    if alias is None:
        alias = official_name
    if official_name not in OFFICIAL_MODEL_NAMES:
        OFFICIAL_MODEL_NAMES.append(official_name)
        MODEL_ALIASES[official_name] = [alias]
    else:
        print(f"Model {official_name} already in the official transformer lens models.")


def expend_tl_cache(cache: KeyValueCache, batch_size: int):
    """
    Expend the cache to the given batch size.
    """
    for entry in cache:
        entry.past_keys = entry.past_keys.expand(batch_size, *entry.past_keys.shape[1:])
        entry.past_values = entry.past_values.expand(
            batch_size, *entry.past_values.shape[1:]
        )
    cache.previous_attention_mask = cache.previous_attention_mask.expand(
        batch_size, *cache.previous_attention_mask.shape[1:]
    )
    return cache


def plot_topk_tokens(
    next_token_probs,
    tokenizer,
    k=4,
    title=None,
    dynamic_size=True,
    dynamic_color_scale=False,
    use_token_ids=False,
    file=None,
):
    """
    Plot the top k tokens for each layer
    :param probs: Probability tensor of shape (num_layers, vocab_size)
    :param k: Number of top tokens to plot
    :param title: Title of the plot
    :param dynamic_size: If True, the size of the plot will be adjusted based on the length of the tokens
    """
    if isinstance(tokenizer, LanguageModel) or isinstance(
        tokenizer, UnifiedTransformer
    ):
        tokenizer = tokenizer.tokenizer
    if next_token_probs.dim() == 1:
        next_token_probs = next_token_probs.unsqueeze(0)
    if next_token_probs.dim() == 2:
        next_token_probs = next_token_probs.unsqueeze(0)
    num_layers = next_token_probs.shape[1]
    max_token_length_sum = 0
    top_token_indices_list = []
    top_probs_list = []
    for probs in next_token_probs:
        top_tokens = th.topk(probs, k=k, dim=-1)
        top_probs = top_tokens.values
        if not use_token_ids:
            top_token_indices = [
                ["'" + tokenizer.convert_ids_to_tokens(t.item()) + "'" for t in l]
                for l in top_tokens.indices
            ]
        else:
            top_token_indices = [[str(t.item()) for t in l] for l in top_tokens.indices]
        top_token_indices_list.append(top_token_indices)
        top_probs_list.append(top_probs)
    for top_token_indices in top_token_indices_list:
        max_token_length_sum += max(
            [len(token) for sublist in top_token_indices for token in sublist]
        )
    has_chinese = any(
        any("\u4e00" <= c <= "\u9fff" for c in token)
        for top_token_indices in top_token_indices_list
        for sublist in top_token_indices
        for token in sublist
    )

    context = (
        mpl.rc_context(rc={"font.sans-serif": [simsun, "Arial"]})
        if has_chinese
        else nullcontext()
    )
    with context:
        if dynamic_size:
            fig, axes = plt.subplots(
                1,
                len(next_token_probs),
                figsize=(max_token_length_sum * k * 0.25, num_layers / 2),
            )
        else:
            fig, axes = plt.subplots(
                1, len(next_token_probs), figsize=(15 * len(next_token_probs), 10)
            )
        if len(next_token_probs) == 1:
            axes = [axes]
        for i, (ax, top_probs, top_token_indices) in enumerate(
            zip(axes, top_probs_list, top_token_indices_list)
        ):
            cmap = sns.diverging_palette(255, 0, as_cmap=True)
            sns_kwargs = {}
            if not dynamic_color_scale:
                sns_kwargs.update(dict(vmin=0, vmax=1, cbar=i == len(axes) - 1))
            sns.heatmap(
                top_probs.detach().numpy(),
                annot=top_token_indices,
                fmt="",
                cmap=cmap,
                linewidths=0.5,
                cbar_kws={"label": "Probability"},
                ax=ax,
                **sns_kwargs,
            )
            ax.set_xlabel("Tokens")
            ax.set_ylabel("Layers")
            ax.set_yticks(np.arange(num_layers) + 0.5, range(num_layers))
        if title is None:
            fig.suptitle(f"Top {k} Tokens Heatmap")
        else:
            fig.suptitle(f"Top {k} Tokens Heatmap - {title}")

        plt.tight_layout()
        if file is not None:
            fig.savefig(file, bbox_inches="tight", dpi=300)
        fig.show()
        plt.show()


def ulist(lst):
    """
    Returns a list with unique elements from the input list.
    """
    return list(dict.fromkeys(lst))

def lfilter(lst, f):
    """
    Returns a list with elements from the input list that satisfy the condition.
    """
    return list(filter(f, lst))

def display_df(df):
    with pd.option_context(
        "display.max_colwidth",
        None,
        "display.max_columns",
        None,
        "display.max_rows",
        None,
    ):
        display(df)

def get_tokenizer(model_or_tokenizer):
    if isinstance(model_or_tokenizer, LanguageModel) or isinstance(
        model_or_tokenizer, UnifiedTransformer
    ):
        return model_or_tokenizer.tokenizer
    return model_or_tokenizer