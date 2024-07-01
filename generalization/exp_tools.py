from __future__ import annotations
from dataclasses import dataclass
import torch as th
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
from nnsight.models.UnifiedTransformer import UnifiedTransformer
from nnsight import LanguageModel
from utils import ulist, get_tokenizer
from typing import Callable
import re
from interventions import logit_lens, patchscope_lens, TargetPrompt
from nnsight_utils import collect_activations, get_num_layers
from typing import Literal, Optional

DATA_PATH = Path(__file__).resolve().parent.parent / "data"


def load_model(model_name: str, trust_remote_code=False, use_tl=False, **kwargs_):
    """
    Load a model into nnsight. If use_tl is True, a TransformerLens model is loaded.
    Default device is "auto" and default torch_dtype is th.float16.
    """
    kwargs = dict(torch_dtype=th.float16, trust_remote_code=trust_remote_code)
    if use_tl:
        kwargs["device"] = "cuda" if th.cuda.is_available() else "cpu"
        kwargs["processing"] = False
        kwargs.update(kwargs_)
        return UnifiedTransformer(model_name, **kwargs)
    else:
        kwargs["device_map"] = "auto"
        tokenizer_kwargs = kwargs_.pop("tokenizer_kwargs", {})
        tokenizer_kwargs.update(
            dict(add_prefix_space=False, trust_remote_code=trust_remote_code)
        )
        kwargs.update(kwargs_)
        return LanguageModel(model_name, tokenizer_kwargs=tokenizer_kwargs, **kwargs)


def get_mean_activations(nn_model, prompts_str, batch_size=32, remote=False):
    dataloader = DataLoader(prompts_str, batch_size=batch_size, shuffle=False)
    acts = []
    for batch in tqdm(dataloader):
        acts.append(collect_activations(nn_model, batch, remote=remote))
    mean_activations = []
    num_layers = get_num_layers(nn_model)
    for layer in range(num_layers):
        mean_activations.append(th.cat([a[layer] for a in acts]).mean(0))
    return mean_activations


def next_token_probs(
    nn_model: UnifiedTransformer | LanguageModel, prompt: str | list[str]
):
    out = nn_model.trace(prompt, trace=False)
    if isinstance(nn_model, LanguageModel):
        out = out.logits
    return out[:, -1].softmax(-1).cpu()


def description_prompt(placeholder="?"):
    return TargetPrompt(
        f"""Jensen Huang is the CEO of NVIDIA, a technology company
New York City is the largest city in the United States
Johnny Depp is a famous actor known for his role in Pirates of the Caribbean
Google is a technology company known for its search engine
Ariana Grande is a famous singer from the United States
Sam Altman is the CEO of OpenAI, a research lab focused on artificial intelligence
The Eiffel Tower is a famous landmark in Paris, France
C++ is a programming language known for its speed and performance
A spoon is a utensil used for eating food
{placeholder}""",
        -1,
    )


def next_token_probs_unsqueeze(nn_model, prompt, scan=True):
    probs = next_token_probs(nn_model, prompt)
    return probs.unsqueeze(1)  # Add a fake layer dimension


@th.no_grad
def run_prompts(
    nn_model,
    prompts,
    batch_size=32,
    get_probs: Callable = None,
    method_kwargs=None,
):
    """
    Run a list of prompts through the model and return the probabilities of the next token for both the target and latent languages.

    Returns:
        Two tensors target_probs and latent_probs of shape (num_prompts, num_layers)
    """
    str_prompts = [prompt.prompt for prompt in prompts]
    dataloader = DataLoader(str_prompts, batch_size=batch_size)
    probs = []
    scan = True
    if get_probs is None:
        get_probs = next_token_probs_unsqueeze
    elif isinstance(get_probs, str):
        raise ValueError("get_probs must be a callable, update your code")  # todo fix and remove
    if method_kwargs is None:
        method_kwargs = {}
    for prompt_batch in tqdm(dataloader, total=len(dataloader), desc="Running prompts"):
        probs.append(get_probs(nn_model, prompt_batch, scan=scan, **method_kwargs))
        scan = False
    probs = th.cat(probs)
    target_probs = []
    latent_probs = {lang: [] for lang in prompts[0].latent_tokens.keys()}
    for i, prompt in enumerate(prompts):
        target_probs.append(probs[i, :, prompt.target_tokens].sum(dim=1))
        for lang, tokens in prompt.latent_tokens.items():
            latent_probs[lang].append(probs[i, :, tokens].sum(dim=1))
    target_probs = th.stack(target_probs).cpu()
    latent_probs = {lang: th.stack(probs).cpu() for lang, probs in latent_probs.items()}
    return target_probs, latent_probs


def filter_prompts_by_prob(prompts, model, treshold=0.3, batch_size=32):
    if treshold <= 0:
        return prompts
    if prompts == []:
        return []
    target_probs, _ = run_prompts(
        model, prompts, batch_size=batch_size, get_probs=next_token_probs_unsqueeze
    )
    return [
        prompt for prompt, prob in zip(prompts, target_probs) if prob.max() >= treshold
    ]


def prompts_to_str(prompts):
    return [prompt.prompt for prompt in prompts]


def prompts_to_df(prompts, tokenizer=None):
    dic = {}
    for i, prompt in enumerate(prompts):
        dic[i] = {
            "prompt": prompt.prompt,
            "target_strings": prompt.target_strings,
        }
        for lang, string in prompt.latent_strings.items():
            dic[i][lang + "_string"] = string
        if tokenizer is not None:
            dic[i]["target_tokens"] = tokenizer.convert_ids_to_tokens(
                prompt.target_tokens
            )
            for lang, tokens in prompt.latent_tokens.items():
                dic[i][lang + "_tokens"] = tokenizer.convert_ids_to_tokens(tokens)
    return pd.DataFrame.from_dict(dic)


def remove_colliding_prompts(prompts, ignore_langs: Optional[str | list[str]] = None):
    return [prompt for prompt in prompts if prompt.has_no_collisions(ignore_langs)]


filter_prompts = remove_colliding_prompts  # todo: remove this alias
