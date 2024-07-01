import papermill as pm
from pathlib import Path
from time import time
from argparse import ArgumentParser
import os
from coolname import generate_slug

root = Path(__file__).parent
notebook_root = root.parent / "notebook-exp"
if __name__ == "__main__":
    os.chdir(root)
    parser = ArgumentParser()
    parser.add_argument("--notebook", "-n", type=str, required=True)
    parser.add_argument(
        "--langs", "-l", type=str, default=["fr", "de", "ru", "en", "zh"], nargs="+"
    )
    parser.add_argument("--model", "-m", type=str)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--trust-remote-code", default=False, action="store_true")
    parser.add_argument("--device", "-d", type=str, default="auto")
    parser.add_argument("--batch-size", "-b", type=int, default=32)
    parser.add_argument(
        "--paper-only",
        "-po",
        action="store_true",
        help="Only run the paper experiment cell instead of all combinations",
    )
    parser.add_argument(
        "--prob-treshold",
        "-fp",
        type=float,
        default=0.3,
        help="Keep only prompts that the model is confident about. Set to 0 to keep all prompts.",
    )

    args, unknown = parser.parse_known_args()
    kwargs = dict(vars(args))
    notebook = kwargs.pop("notebook")
    print(f"Running {notebook} with {kwargs}")
    save_path = root / "results" / notebook
    save_path.mkdir(exist_ok=True, parents=True)
    source_notebook_path = notebook_root / f"{notebook}.ipynb"
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
    except (pm.PapermillExecutionError, KeyboardInterrupt) as e:
        print(e)
        if isinstance(e, pm.PapermillExecutionError):
            print("Error in notebook")
        delete = input(f"Delete notebook {target_notebook_path}? (y/n)")
        if delete == "y":
            target_notebook_path.unlink()
        else:
            print(f"Notebook saved")
