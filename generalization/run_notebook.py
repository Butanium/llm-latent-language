import papermill as pm
from pathlib import Path
from time import time
from argparse import ArgumentParser
import os
from coolname import generate_slug

root = Path(__file__).parent
if __name__ == "__main__":
    os.chdir(root)
    parser = ArgumentParser()
    parser.add_argument("--notebook", "-n", type=str, default="translation")
    parser.add_argument("--check_translation_performance", type=bool, default=False)
    parser.add_argument(
        "--langs", "-l", type=str, default=["fr", "de", "ru", "en", "zh"], nargs="+"
    )
    parser.add_argument("--batch-size", "-b", type=int, default=32)
    parser.add_argument("--model", "-m", type=str)
    parser.add_argument("--device", "-d", type=str, default="auto")
    parser.add_argument("--method", "-mt", type=str, default="logit_lens")
    parser.add_argument("--thinking-langs", "-t", type=str, default=["en"], nargs="+")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--trust-remote-code", default=False, action="store_true")
    parser.add_argument("--llama", default=False, action="store_true")
    parser.add_argument("--paper-only", "-po", action="store_true")

    args, unknown = parser.parse_known_args()
    kwargs = dict(vars(args))
    kwargs.pop("llama")
    notebook = kwargs.pop("notebook")
    suf = "_llama" if args.llama else ""
    print(f"Running {notebook} with {kwargs}")
    save_path = root / "results" / notebook
    save_path.mkdir(exist_ok=True, parents=True)
    source_notebook_path = root / f"{notebook}{suf}.ipynb"
    exp_id = str(int(time())) + "_" + generate_slug(2)
    target_notebook_path = save_path / (
        args.model.replace("/", "_") + f"_{exp_id}.ipynb"
    )
    kwargs["exp_id"] = exp_id
    print(f"Saving to {target_notebook_path}")
    kwargs["extra_args"] = unknown
    try:
        pm.execute_notebook(
            source_notebook_path,
            target_notebook_path,
            parameters=kwargs,
        )
    except pm.PapermillExecutionError as e:
        print(e)
        print("Error in notebook")
        delete = input(f"Delete notebook {target_notebook_path}? (y/n)")
        if delete == "y":
            target_notebook_path.unlink()
        else:
            print(f"Notebook saved")
