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
from warnings import warn
from typing import Optional, Union

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
    try:
        start = zh_char.encode().__str__()[2:-1].split("\\x")[1]
    except IndexError:
        warn(
            f"Could not encode character {zh_char}, probably not a valid special character"
        )
        return None
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


@dataclass
class PatchScopePrompt:
    prompt: str
    index_to_patch: int


def repeat_prompt(
    nn_model=None, words=None, rel=" ", sep="\n", placeholder="?"
) -> PatchScopePrompt:
    """
    Prompt used in the patchscopes paper to predict the next token.
    https://github.com/PAIR-code/interpretability/blob/master/patchscopes/code/next_token_prediction.ipynb
    """
    if words is None:
        words = [
            "king",
            "1135",
            "hello",
        ]
    assert nn_model is None or (
        len(nn_model.tokenizer.tokenize(placeholder)) == 1
    ), "Using a placeholder that is not a single token sounds like a bad idea"
    prompt = sep.join([w + rel + w for w in words]) + sep + placeholder
    index_to_patch = -1
    return PatchScopePrompt(prompt, index_to_patch)


def description_prompt(placeholder="?"):
    return PatchScopePrompt(
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


@th.no_grad
def patchscope_lens_llama(
    nn_model: LanguageModel,
    prompts: list[str] | str,
    patch_prompt: PatchScopePrompt | None = None,
    scan=True,
    remote=False,
):
    """
    Replace the hidden state of the patch_prompt.index_to_patch token in the patch_prompt.prompt with the hidden state of the last token of each prompt at each layer.
    Returns the probabilities of the next token in patch_prompt for each prompt for each layer intervention.
    Args:
        nn_model: The NNSight LanguageModel with llama architecture
        prompts: List of prompts or a single prompt to get the hidden states of the last token
        patch_prompt: A PatchScopePrompt object containing the prompt to patch and the index of the token to patch
        scan: If looping over this function, set to False after the first call to speed up subsequent calls
        remote: If True, the function will run on the nndif server. See `nnsight.net/status` to check which models are available.
    Returns:
        A tensor of shape (num_prompts, num_layers, vocab_size) containing the probabilities
        of the next token for each prompt at each layer. Tensor is on the CPU.
    """
    if patch_prompt is None:
        patch_prompt = repeat_prompt()
    nn_model.eval()
    tok_prompts = nn_model.tokenizer(prompts, return_tensors="pt", padding=True)
    # Todo?: This is a hacky way to get the last token index but it works for both left and right padding
    last_token_index = (
        tok_prompts.attention_mask.flip(1).cumsum(1).bool().int().sum(1).sub(1)
    )
    # Collect the hidden states of the last token of each prompt at each layer
    if isinstance(prompts, str):
        prompts = [prompts]
    probs_l = []
    with nn_model.trace(remote=remote) as tracer:
        with tracer.invoke(prompts, scan=scan):
            hiddens = [
                layer.output[0][
                    th.arange(len(tok_prompts.input_ids)),
                    last_token_index,
                ].save()
                for layer in nn_model.model.layers
            ]
        if isinstance(prompts, str):
            prompts = [prompts]
        # For each prompt, we will do n_layers patching so we need to expand the cache accordingly
        n_layers = len(nn_model.model.layers)
        # Collect the patch activations for each prompt at each layer
        for layer in range(n_layers):
            with tracer.invoke(
                [patch_prompt.prompt] * len(prompts), scan=(scan and layer == 0)
            ):
                nn_model.model.layers[layer].output[0][
                    :, patch_prompt.index_to_patch
                ] = hiddens[layer]
                output = nn_model.lm_head.output
                probs = output[:, -1, :].softmax(-1).cpu().save()
                probs_l.append(probs)
    probs = th.cat(probs_l, dim=0)
    return probs.reshape(n_layers, len(prompts), -1).transpose(0, 1)


@th.no_grad
def patchscope_generate_llama(
    nn_model: LanguageModel,
    prompts: list[str] | str,
    patch_prompt: PatchScopePrompt,
    max_length=50,
    layers=None,
    stopping_criteria=None,
    scan=True,
    remote=False,
    max_batch_size=32,
):
    """
    Replace the hidden state of the patch_prompt.index_to_patch token in the patch_prompt.prompt with the hidden state of the last token of each prompt at each layer.
    Returns the probabilities of the next token in patch_prompt for each prompt for each layer intervention.
    Args:
        nn_model: The NNSight LanguageModel with llama architecture
        prompts: List of prompts or a single prompt to get the hidden states of the last token
        patch_prompt: A PatchScopePrompt object containing the prompt to patch and the index of the token to patch
        layers: List of layers to intervene on. If None, all layers are intervened on.
        stopping_criteria: The HF stopping criteria which stops generation when it is met for ALL prompts
        scan: If looping over this function, set to False after the first call to speed up subsequent calls
        remote: If True, the function will run on the nndif server. See `nnsight.net/status` to check which models are available.
        max_batch_size: The maximum number of prompts to intervene on at once.

    Returns:
        A tensor of shape (num_prompts, num_layers, vocab_size) containing the probabilities
        of the next token for each prompt at each layer. Tensor is on the CPU.
    """
    if isinstance(prompts, str):
        prompts = [prompts]
    if len(prompts) > max_batch_size:
        warn(
            f"Number of prompts ({len(prompts)}) exceeds max_batch_size ({max_batch_size}). This may cause memory errors."
        )
    nn_model.eval()
    n_layers = len(nn_model.model.layers)
    if layers is None:
        layers = list(range(n_layers))
    tok_prompts = nn_model.tokenizer(prompts, return_tensors="pt", padding=True)
    # Todo?: This is a hacky way to get the last token index but it works for both left and right padding
    last_token_index = (
        tok_prompts.attention_mask.flip(1).cumsum(1).bool().int().sum(1).sub(1)
    )
    # Collect the hidden states of the last token of each prompt at each layer
    with nn_model.trace(prompts, scan=scan, remote=remote):
        hiddens = [
            layer.output[0][
                th.arange(len(tok_prompts.input_ids)),
                last_token_index,
            ].save()
            for layer in nn_model.model.layers
        ]
    generations = {}
    gen_kwargs = dict(
        remote=remote, max_new_tokens=max_length, stopping_criteria=stopping_criteria
    )
    layer_loader = DataLoader(layers, batch_size=max(max_batch_size // len(prompts), 1))
    for layer_batch in layer_loader:
        with nn_model.generate(**gen_kwargs) as tracer:
            if isinstance(prompts, str):
                prompts = [prompts]
            for layer in layer_batch:
                layer = layer.item()
                with tracer.invoke(
                    [patch_prompt.prompt] * len(prompts), scan=(scan and layer == 0)
                ):
                    nn_model.model.layers[layer].output[0][
                        :, patch_prompt.index_to_patch
                    ] = hiddens[layer]
                    gen = nn_model.generator.output.save()
                    generations[layer] = gen
    for k, v in generations.items():
        generations[k] = v.cpu()
    return generations


@th.no_grad
def patchscope_lens(
    nn_model: UnifiedTransformer,
    prompts: list[str] | str,
    patch_prompt: PatchScopePrompt | None = None,
    scan=True,
    remote=False,
):
    """
    Replace the hidden state of the patch_prompt.index_to_patch token in the patch_prompt.prompt with the hidden state of the last token of each prompt at each layer.
    Returns the probabilities of the next token in patch_prompt for each prompt for each layer intervention.
    Args:
        nn_model: The NNSight TL model
        prompts: List of prompts or a single prompt to get the hidden states of the last token
        patch_prompt: A PatchScopePrompt object containing the prompt to patch and the index of the token to patch
        scan: If looping over this function, set to False after the first call to speed up subsequent calls
        remote: If True, the function will run on the nndif server. See `nnsight.net/status` to check which models are available.
    Returns:
        A tensor of shape (num_prompts, num_layers, vocab_size) containing the probabilities
        of the next token for each prompt at each layer. Tensor is on the CPU.
    """
    if patch_prompt is None:
        patch_prompt = repeat_prompt()
    nn_model.eval()
    tok_prompts = nn_model.tokenizer(prompts, return_tensors="pt", padding=True)
    # Todo?: This is a hacky way to get the last token index but it works for both left and right padding
    last_token_index = (
        tok_prompts.attention_mask.flip(1).cumsum(1).bool().int().sum(1).sub(1)
    )
    # Collect the hidden states of the last token of each prompt at each layer
    if isinstance(prompts, str):
        prompts = [prompts]
    probs_l = []
    with nn_model.trace(prompts, scan=scan, remote=remote) as tracer:
        hiddens = [
            layer.output[
                th.arange(len(tok_prompts.input_ids)),
                last_token_index,
            ].save()
            for layer in nn_model.blocks
        ]
        # For each prompt, we will do n_layers patching so we need to expand the cache accordingly
    n_layers = len(nn_model.blocks)
    # Collect the patch activations for each prompt at each layer
    for layer in range(n_layers):
        with nn_model.trace(
            [patch_prompt.prompt] * len(prompts),
            scan=layer == 0,
            remote=remote,
        ):
            nn_model.blocks[layer].output[:, patch_prompt.index_to_patch] = hiddens[
                layer
            ]
            x = nn_model.blocks[layer].output[:, patch_prompt.index_to_patch].save()
            output = nn_model.unembed.output
            probs = output[:, -1, :].softmax(-1).cpu().save()
            probs_l.append(probs)
        # print(x.shape, hi.shape)
    probs = th.cat(probs_l, dim=0)
    return probs.reshape(n_layers, len(prompts), -1).transpose(0, 1)


@th.no_grad
def patchscope_generate(
    nn_model: UnifiedTransformer,
    prompts: list[str] | str,
    patch_prompt: PatchScopePrompt,
    max_length=50,
    layers=None,
    stop_tokens: Optional[Union[int, list[int]]] = None,
    scan=True,
    remote=False,
    max_batch_size=32,
):
    """
    Replace the hidden state of the patch_prompt.index_to_patch token in the patch_prompt.prompt with the hidden state of the last token of each prompt at each layer.
    Returns the probabilities of the next token in patch_prompt for each prompt for each layer intervention.
    Args:
        nn_model: The NNSight TL model
        prompts: List of prompts or a single prompt to get the hidden states of the last token
        patch_prompt: A PatchScopePrompt object containing the prompt to patch and the index of the token to patch
        layers: List of layers to intervene on. If None, all layers are intervened on.
        stop_tokens: The tokens to stop generation at. If None, generation will stop at the end of the prompt.
        scan: If looping over this function, set to False after the first call to speed up subsequent calls
        remote: If True, the function will run on the nndif server. See `nnsight.net/status` to check which models are available.
        max_batch_size: The maximum number of prompts to intervene on at once.

    Returns:
        A tensor of shape (num_prompts, num_layers, vocab_size) containing the probabilities
        of the next token for each prompt at each layer. Tensor is on the CPU.
    """
    if isinstance(prompts, str):
        prompts = [prompts]
    if len(prompts) > max_batch_size:
        warn(
            f"Number of prompts ({len(prompts)}) exceeds max_batch_size ({max_batch_size}). This may cause memory errors."
        )
    if isinstance(stop_tokens, int):
        stop_tokens = [stop_tokens]
    if stop_tokens is not None:
        stop_tokens.append(nn_model.tokenizer.eos_token_id)
    nn_model.eval()
    n_layers = len(nn_model.blocks)
    if layers is None:
        layers = list(range(n_layers))
    tok_prompts = nn_model.tokenizer(prompts, return_tensors="pt", padding=True)
    # Todo?: This is a hacky way to get the last token index but it works for both left and right padding
    last_token_index = (
        tok_prompts.attention_mask.flip(1).cumsum(1).bool().int().sum(1).sub(1)
    )
    # Collect the hidden states of the last token of each prompt at each layer
    with nn_model.trace(prompts, scan=scan, remote=remote):
        hiddens = [
            layer.output[
                th.arange(len(tok_prompts.input_ids)),
                last_token_index,
            ].save()
            for layer in nn_model.blocks
        ]
    generations = {}
    gen_kwargs = dict(
        remote=remote, max_new_tokens=max_length, eos_token_id=stop_tokens
    )
    layer_loader = DataLoader(layers, batch_size=max(max_batch_size // len(prompts), 1))
    for layer_batch in layer_loader:
        with nn_model.generate(**gen_kwargs) as tracer:
            if isinstance(prompts, str):
                prompts = [prompts]
            for layer in layer_batch:
                layer = layer.item()
                with tracer.invoke(
                    [patch_prompt.prompt] * len(prompts), scan=(scan and layer == 0)
                ):
                    nn_model.blocks[layer].output[:, patch_prompt.index_to_patch] = (
                        hiddens[layer]
                    )
                    gen = nn_model.generator.output.save()
                    generations[layer] = gen
    for k, v in generations.items():
        generations[k] = v.cpu()
    return generations


method_to_fn = {
    "logit_lens": logit_lens,
    "logit_lens_llama": logit_lens_llama,
    "patchscope_lens": patchscope_lens,
    "patchscope_lens_llama": patchscope_lens_llama,
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
