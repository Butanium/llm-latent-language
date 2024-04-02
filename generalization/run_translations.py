import papermill as pm
from pathlib import Path
from time import time
from argparse import ArgumentParser
import os


models = [
    "gpt2",
    "google/gemma-2b",
    "croissantllm/CroissantLLMBase",
]
thinking_langs = [["en"], ["en"], ["fr", "en"]]

root = Path(__file__).parent
if __name__ == "__main__":
    os.chdir(root)
    time = str(int(time()))
    # check_translation_performance = True
    parser = ArgumentParser()
    parser.add_argument("--check_translation_performance", type=bool, default=True)
    parser.add_argument("--langs", type=list, default=["fr", "de", "ru", "en", "zh"])
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    for model, langs in zip(models, thinking_langs):
        pm.execute_notebook(
            root / "translation.ipynb",
            root / "results" / (model.replace("/", "_") + f"_{time}.ipynb"),
            parameters=dict(model=model, thinking_langs=langs, **vars(args)),
        )
