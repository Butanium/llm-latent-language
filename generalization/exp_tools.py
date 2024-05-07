from __future__ import annotations
from dataclasses import dataclass
import torch as th
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
from nnsight.models.UnifiedTransformer import UnifiedTransformer
from nnsight import LanguageModel
from transformer_lens import HookedTransformerKeyValueCache as KeyValueCache
from utils import expend_tl_cache
from typing import Callable
from nnsight import logger

logger.disabled = True
from typing import Literal

DATA_PATH = Path(__file__).resolve().parent.parent / "data"


def load_lang(lang):
    path = DATA_PATH / "langs" / lang / "clean.csv"
    return pd.read_csv(path)


def token_prefixes(token_str: str):
    n = len(token_str)
    tokens = [token_str[:i] for i in range(1, n + 1)]
    return tokens


SPACE_TOKENS = ["▁", "Ġ"]


def add_spaces(tokens):
    return sum([[s + token for token in tokens] for s in SPACE_TOKENS], []) + tokens


# TODO?: Add capitalization


def unicode_prefix_tokid(zh_char, tok_vocab):
    start = zh_char.encode().__str__()[2:-1].split("\\x")[1]
    unicode_format = "<0x%s>"
    start_key = unicode_format % start.upper()
    if start_key in tok_vocab:
        return tok_vocab[start_key]
    return None


def process_tokens(words: str | list[str], tok_vocab, lang=None):
    if isinstance(words, str):
        words = [words]
    final_tokens = []
    for word in words:
        with_prefixes = token_prefixes(word)
        with_spaces = add_spaces(with_prefixes)
        for cap_word in with_spaces:
            if cap_word in tok_vocab:
                final_tokens.append(tok_vocab[cap_word])
        if lang in ["zh", "ru"]:
            tokid = unicode_prefix_tokid(word, tok_vocab)
            if tokid is not None:
                final_tokens.append(tokid)
    return list(set(final_tokens))


@th.no_grad
def logit_lens(
    nn_model: UnifiedTransformer, prompts: list[str] | str, scan=True, remote=False
):
    """
    Get the probabilities of the next token for the last token of each prompt at each layer using the logit lens.

    Args:
        nn_model: NNSight LanguageModel object
        prompts: List of prompts or a single prompt

    Returns:
        A tensor of shape (num_prompts, num_layers, vocab_size) containing the probabilities
        of the next token for each prompt at each layer. Tensor is on the CPU.
    """
    nn_model.eval()
    tok_prompts = nn_model.tokenizer(prompts, return_tensors="pt", padding=True)
    # Todo?: This is a hacky way to get the last token index but it works for both left and right padding
    last_token_index = (
        tok_prompts.attention_mask.flip(1).cumsum(1).bool().int().sum(1).sub(1)
    )
    with nn_model.trace(prompts, scan=scan, remote=remote) as tracer:
        hiddens_l = [
            layer.output[
                th.arange(len(tok_prompts.input_ids), device=layer.output.device),
                last_token_index.to(layer.output.device),
            ].unsqueeze(1)
            for layer in nn_model.blocks
        ]
        probs_l = []
        for hiddens in hiddens_l:
            ln_out = nn_model.ln_final(hiddens)
            logits = nn_model.unembed(ln_out)
            probs_l.append(logits.squeeze(1).softmax(-1).cpu())
        probs = th.stack(probs_l).transpose(0, 1).save()
    return probs


@th.no_grad
def logit_lens_llama(
    nn_model: LanguageModel, prompts: list[str] | str, scan=True, remote=False
):
    """
    Same as logit_lens but for Llama models directly instead of Transformer_lens models.
    Get the probabilities of the next token for the last token of each prompt at each layer using the logit lens.

    Args:
        nn_model: NNSight LanguageModel object
        prompts: List of prompts or a single prompt

    Returns:
        A tensor of shape (num_prompts, num_layers, vocab_size) containing the probabilities
        of the next token for each prompt at each layer. Tensor is on the CPU.
    """
    nn_model.eval()
    tok_prompts = nn_model.tokenizer(prompts, return_tensors="pt", padding=True)
    # Todo?: This is a hacky way to get the last token index but it works for both left and right padding
    last_token_index = (
        tok_prompts.attention_mask.flip(1).cumsum(1).bool().int().sum(1).sub(1)
    )
    with nn_model.trace(prompts, scan=scan, remote=remote) as tracer:
        hiddens_l = [
            layer.output[0][
                th.arange(len(tok_prompts.input_ids)),
                last_token_index,
            ]
            for layer in nn_model.model.layers
        ]
        probs_l = []
        for hiddens in hiddens_l:
            ln_out = nn_model.model.norm(hiddens)
            logits = nn_model.lm_head(ln_out)
            probs = logits.softmax(-1).cpu()
            probs_l.append(probs)
        probs = th.stack(probs_l).transpose(0, 1).save()
    return probs


def patchscope_prompt(words, rel="→", sep="; "):
    return sep.join([w + rel + w for w in words]) + sep


""" TODO: Fix implementation
@th.no_grad
def patchscope(
    nn_model: UnifiedTransformer,
    prompts: list[str] | str,
    patch_prompt=None,
    scan=True,
    rel="→",
    sep="; ",
    placeholder="_",
):
    assert (
        len(nn_model.tokenizer(placeholder, add_special_tokens=False).input_ids) == 1
    ), "Placeholder must be a single token"
    nn_model.eval()
    if patch_prompt is None:
        patch_prompt = patchscope_prompt(["hello", "123", "cow"], rel=rel, sep=sep)
    tok_prompts = nn_model.tokenizer(prompts, return_tensors="pt", padding=True)
    # Todo?: This is a hacky way to get the last token index but it works for both left and right padding
    last_token_index = (
        tok_prompts.attention_mask.flip(1).cumsum(1).bool().int().sum(1).sub(1)
    )
    # Collect the hidden states of the last token of each prompt at each layer
    with nn_model.trace(prompts, scan=scan) as tracer:
        hiddens_l = [
            layer.output[
                th.arange(len(tok_prompts.input_ids), device=layer.output.device),
                last_token_index.to(layer.output.device),
            ].unsqueeze(1)
            for layer in nn_model.blocks
        ]
        hiddens = th.cat(hiddens_l, dim=1).save()
    kv_cache = KeyValueCache.init_cache(nn_model.cfg, nn_model.cfg.device)
    # Precompute the patch prompt activations
    with nn_model.trace(patch_prompt, past_kv_cache=kv_cache, scan=scan):
        pass
    kv_cache.freeze()
    if isinstance(prompts, str):
        prompts = [prompts]
    # For each prompt, we will do n_layers patching so we need to expand the cache accordingly
    n_layers = nn_model.cfg.n_layers
    expend_tl_cache(kv_cache, len(prompts) * n_layers)
    # Collect the patch activations for each prompt at each layer
    with nn_model.trace(
        [placeholder + rel] * (len(prompts) * n_layers), past_kv_cache=kv_cache,
    ) as tracer:
        for layer in range(n_layers):
            nn_model.blocks[layer].output[
                layer * len(prompts) : (layer + 1) * len(prompts), 0
            ] = hiddens[:, layer]
        output = nn_model.unembed.output
        probs = output[:, -1, :].softmax(-1).cpu().save()
    return probs.reshape(n_layers, len(prompts), -1).transpose(0, 1)
 """

method_to_fn = {
    "logit_lens": logit_lens,
    "logit_lens_llama": logit_lens_llama,
    # "patchscope": patchscope,
}


@dataclass
class Prompt:
    prompt: str
    target_tokens: list[int]
    latent_tokens: dict[str, list[int]]
    target_string: str
    latent_strings: dict[str, str | list[str]]

    @th.no_grad
    def run(
        self,
        nn_model,
        method: Literal["logit_lens", "patchscope", "logit_lens_llama"] = "logit_lens",
    ):
        """
        Run the prompt through the model and return the probabilities of the next token for both the target and latent languages.
        """
        get_probs = method_to_fn[method]
        probs = get_probs(nn_model, self.prompt)
        target_probs = probs[:, :, self.target_tokens].sum(dim=2)
        latent_probs = {
            lang: probs[:, :, tokens].sum(dim=2)
            for lang, tokens in self.latent_tokens.items()
        }
        return target_probs.cpu(), latent_probs.cpu()


@th.no_grad
def run_prompts(
    nn_model,
    prompts,
    batch_size=32,
    method: (
        Literal["logit_lens", "patchscope", "logit_lens_llama"] | Callable
    ) = "logit_lens",
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
    if isinstance(method, str):
        get_probs = method_to_fn[method]
    else:
        get_probs = method
    for prompt_batch in tqdm(dataloader, total=len(dataloader), desc="Running prompts"):
        probs.append(get_probs(nn_model, prompt_batch, scan=scan))
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
