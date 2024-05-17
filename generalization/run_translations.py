import papermill as pm
from pathlib import Path
from time import time
from argparse import ArgumentParser
import os

root = Path(__file__).parent
if __name__ == "__main__":
    os.chdir(root)
    time = str(int(time()))
    parser = ArgumentParser()
    parser.add_argument("--check_translation_performance", type=bool, default=False)
    parser.add_argument("--langs", "-l", type=str, default=["fr", "de", "ru", "en", "zh"], nargs="+")
    parser.add_argument("--batch-size", "-b", type=int, default=32)
    parser.add_argument("--model", "-m", type=str)
    parser.add_argument("--method", "-mt", type=str, default="logit_lens")
    parser.add_argument("--thinking-langs", "-t", type=str, default=["en"], nargs="+")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--trust_remote_code", default=False, action="store_true")
    parser.add_argument("--llama", default=False, action="store_true")
    args = parser.parse_args()
    kwargs = dict(vars(args))
    kwargs.pop("llama")
    suf = "_llama" if args.llama else ""
    pm.execute_notebook(
        root / f"translation{suf}.ipynb",
        root / "results" / (args.model.replace("/", "_") + f"_{time}.ipynb"),
        parameters=kwargs,
    )
