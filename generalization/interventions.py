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
from utils import expend_tl_cache, ulist
from typing import Union
from warnings import warn
from typing import Optional, Union, Callable
import re
from nnsight_utils import (
    get_layer_output,
    get_attention_output,
    get_logits,
    get_probs,
    collect_hidden_states,
    get_num_layers,
    NNLanguageModel,
    GetModuleOutput,
)


@dataclass
class TargetPrompt:
    prompt: str
    index_to_patch: int


def repeat_prompt(
    nn_model=None, words=None, rel=" ", sep="\n", placeholder="?"
) -> TargetPrompt:
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
    return TargetPrompt(prompt, index_to_patch)


@dataclass
class TargetPromptBatch:
    prompts: list[str]
    index_to_patch: th.Tensor

    @classmethod
    def from_patchscope_prompts(cls, prompts_: list[TargetPrompt], tokenizer=None):
        prompts = [p.prompt for p in prompts_]
        index_to_patch = th.tensor([p.index_to_patch for p in prompts_])
        if index_to_patch.min() < 0:
            if tokenizer is None:
                raise ValueError(
                    "If using negative index_to_patch, a tokenizer must be provided"
                )
        return cls(prompts, index_to_patch)

    @classmethod
    def from_patchscope_prompt(cls, prompt: TargetPrompt, batch_size: int):
        prompts = [prompt.prompt] * batch_size
        index_to_patch = th.tensor([prompt.index_to_patch] * batch_size)
        return cls(prompts, index_to_patch)

    @classmethod
    def from_prompts(
        cls, prompts: str | list[str], index_to_patch: int | list[int] | th.Tensor
    ):
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(index_to_patch, int):
            index_to_patch = th.tensor([index_to_patch] * len(prompts))
        elif isinstance(index_to_patch, list):
            index_to_patch = th.tensor(index_to_patch)
        elif not isinstance(index_to_patch, th.Tensor):
            raise ValueError(
                f"index_to_patch must be an int, a list of ints or a tensor, got {type(index_to_patch)}"
            )
        return cls(prompts, index_to_patch)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return TargetPrompt(self.prompts[idx], self.index_to_patch[idx])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @staticmethod
    def auto(
        target_prompt: str | TargetPrompt | list[TargetPrompt] | TargetPromptBatch,
        batch_size: int,
    ):
        if isinstance(target_prompt, TargetPrompt):
            target_prompt = TargetPromptBatch.from_patchscope_prompt(
                target_prompt, batch_size
            )
        elif isinstance(target_prompt, list):
            target_prompt = TargetPromptBatch.from_patchscope_prompts(target_prompt)
        elif not isinstance(target_prompt, TargetPromptBatch):
            raise ValueError(
                f"patch_prompts must be a str, a TargetPrompt, a list of TargetPrompt or a TargetPromptBatch, got {type(target_prompt)}"
            )
        return target_prompt


@th.no_grad
def patchscope_lens(
    nn_model: UnifiedTransformer,
    source_prompts: list[str] | str,
    target_patch_prompts: (
        TargetPromptBatch | list[TargetPrompt] | TargetPrompt | None
    ) = None,
    scan=True,
    remote=False,
):
    """
    Replace the hidden state of the patch_prompt.index_to_patch token in the patch_prompt.prompt with the hidden state of the last token of each prompt at each layer.
    Returns the probabilities of the next token in patch_prompt for each prompt for each layer intervention.
    Args:
        nn_model: The NNSight TL model
        source_prompts: List of prompts or a single prompt to get the hidden states of the last token
        target_patch_prompt: A TargetPrompt object containing the prompt to patch and the index of the token to patch
        scan: If looping over this function, set to False after the first call to speed up subsequent calls
        remote: If True, the function will run on the nndif server. See `nnsight.net/status` to check which models are available.
    Returns:
        A tensor of shape (num_prompts, num_layers, vocab_size) containing the probabilities
        of the next token for each prompt at each layer. Tensor is on the CPU.
    """
    if target_patch_prompts is None:
        target_patch_prompts = repeat_prompt()
    target_patch_prompts = TargetPromptBatch.auto(
        target_patch_prompts, len(source_prompts)
    )
    if len(target_patch_prompts) != len(source_prompts):
        raise ValueError(
            f"Number of prompts ({len(source_prompts)}) does not match number of patch prompts ({len(target_patch_prompts)})"
        )
    hiddens = collect_hidden_states(nn_model, source_prompts, remote=remote)
    probs_l = []
    n_layers = len(nn_model.blocks)
    # Collect the patch activations for each prompt at each layer
    for layer in range(n_layers):
        with nn_model.trace(
            target_patch_prompts.prompts,
            scan=layer == 0,
            remote=remote,
        ):
            nn_model.blocks[layer].output[
                th.arange(len(source_prompts)), target_patch_prompts.index_to_patch
            ] = hiddens[layer]
            output = nn_model.unembed.output
            probs = output[:, -1, :].softmax(-1).cpu().save()
            probs_l.append(probs)
    probs = th.cat(probs_l, dim=0)
    return probs.reshape(n_layers, len(source_prompts), -1).transpose(0, 1)


@th.no_grad
def patchscope_generate(
    nn_model: UnifiedTransformer,
    prompts: list[str] | str,
    patch_prompt: TargetPrompt,
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
        patch_prompt: A TargetPrompt object containing the prompt to patch and the index of the token to patch
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
    if layers is None:
        layers = list(range(len(nn_model.blocks)))
    hiddens = collect_hidden_states(nn_model, prompts, remote=remote, layers=layers)
    generations = {}
    gen_kwargs = dict(
        remote=remote, max_new_tokens=max_length, eos_token_id=stop_tokens
    )
    layer_loader = DataLoader(layers, batch_size=max(max_batch_size // len(prompts), 1))
    for layer_batch in layer_loader:
        with nn_model.generate(**gen_kwargs) as tracer:
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


@th.no_grad
def patchscope_lens_llama(
    nn_model: LanguageModel,
    source_prompts: list[str] | str,
    target_patch_prompts: (
        TargetPromptBatch | list[TargetPrompt] | TargetPrompt | None
    ) = None,
    scan=True,
    remote=False,
):
    """
    Replace the hidden state of the patch_prompt.index_to_patch token in the patch_prompt.prompt with the hidden state of the last token of each prompt at each layer.
    Returns the probabilities of the next token in patch_prompt for each prompt for each layer intervention.
    Args:
        nn_model: The NNSight LanguageModel with llama architecture
        source_prompts: List of prompts or a single prompt to get the hidden states of the last token
        target_patch_prompt: A TargetPrompt object containing the prompt to patch and the index of the token to patch
        scan: If looping over this function, set to False after the first call to speed up subsequent calls
        remote: If True, the function will run on the nndif server. See `nnsight.net/status` to check which models are available.
    Returns:
        A tensor of shape (num_prompts, num_layers, vocab_size) containing the probabilities
        of the next token for each prompt at each layer. Tensor is on the CPU.
    """
    if isinstance(source_prompts, str):
        source_prompts = [source_prompts]
    if target_patch_prompts is None:
        target_patch_prompts = repeat_prompt()
    if isinstance(target_patch_prompts, TargetPrompt):
        target_patch_prompts = TargetPromptBatch.from_patchscope_prompt(
            target_patch_prompts, len(source_prompts)
        )
    elif isinstance(target_patch_prompts, list):
        target_patch_prompts = TargetPromptBatch.from_patchscope_prompts(
            target_patch_prompts
        )
    elif not isinstance(target_patch_prompts, TargetPromptBatch):
        raise ValueError(
            f"patch_prompts must be a TargetPrompt, a list of TargetPrompt or a TargetPromptBatch, got {type(target_patch_prompts)}"
        )
    if len(target_patch_prompts) != len(source_prompts):
        raise ValueError(
            f"Number of prompts ({len(source_prompts)}) does not match number of patch prompts ({len(target_patch_prompts)})"
        )
    hiddens = collect_hidden_states(nn_model, source_prompts, remote=remote)
    probs_l = []
    n_layers = len(nn_model.model.layers)
    # Collect the patch activations for each prompt at each layer
    for layer in range(n_layers):
        with nn_model.trace(
            target_patch_prompts.prompts,
            scan=layer == 0,
            remote=remote,
        ):
            nn_model.model.layers[layer].output[0][
                th.arange(len(source_prompts)), target_patch_prompts.index_to_patch
            ] = hiddens[layer]
            output = nn_model.lm_head.output
            probs = output[:, -1, :].softmax(-1).cpu().save()
            probs_l.append(probs)
    probs = th.cat(probs_l, dim=0)
    return probs.reshape(n_layers, len(source_prompts), -1).transpose(0, 1)


@th.no_grad
def patchscope_generate_llama(
    nn_model: LanguageModel,
    prompts: list[str] | str,
    target_patch_prompt: TargetPrompt,
    max_length: int = 50,
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
        patch_prompt: A TargetPrompt object containing the prompt to patch and the index of the token to patch
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
    hiddens = collect_hidden_states(nn_model, prompts, remote=remote, layers=layers)
    generations = {}
    gen_kwargs = dict(
        remote=remote, max_new_tokens=max_length, stopping_criteria=stopping_criteria
    )
    layer_loader = DataLoader(layers, batch_size=max(max_batch_size // len(prompts), 1))
    for layer_batch in layer_loader:
        with nn_model.generate(**gen_kwargs) as tracer:
            for layer in layer_batch:
                layer = layer.item()
                with tracer.invoke(
                    [target_patch_prompt.prompt] * len(prompts),
                    scan=(scan and layer == 0),
                ):
                    nn_model.model.layers[layer].output[0][
                        :, target_patch_prompt.index_to_patch
                    ] = hiddens[layer]
                    gen = nn_model.generator.output.save()
                    generations[layer] = gen
    for k, v in generations.items():
        generations[k] = v.cpu()
    return generations


@th.no_grad
def patchscope_intervention_llama(
    nn_model: LanguageModel,
    source_prompts: list[str] | str,
    target_patch_prompts: TargetPromptBatch | list[TargetPrompt] | TargetPrompt,
    source_layer: int,
    target_layer: int,
    start_skip: Optional[int] = None,
    end_skip: Optional[int] = None,
    scan=True,
    remote=False,
):
    """
    Replace the hidden state of the patch_prompt.index_to_patch token in the patch_prompt.prompt with the hidden state of the last token of each prompt at each layer.
    Returns the probabilities of the next token in patch_prompt for each prompt for each layer intervention.
    Args:
        nn_model: The NNSight LanguageModel with llama architecture
        source_prompts: List of prompts or a single prompt to get the hidden states of the last token
        target_patch_prompt: A TargetPrompt object containing the prompt to patch and the index of the token to patch
        scan: If looping over this function, set to False after the first call to speed up subsequent calls
        remote: If True, the function will run on the nndif server. See `nnsight.net/status` to check which models are available.
    Returns:
        A tensor of shape (num_prompts, num_layers, vocab_size) containing the probabilities
        of the next token for each prompt at each layer. Tensor is on the CPU.
    """
    if isinstance(source_prompts, str):
        source_prompts = [source_prompts]
    if target_patch_prompts is None:
        target_patch_prompts = repeat_prompt()
    if isinstance(target_patch_prompts, TargetPrompt):
        target_patch_prompts = TargetPromptBatch.from_patchscope_prompt(
            target_patch_prompts, len(source_prompts)
        )
    elif isinstance(target_patch_prompts, list):
        target_patch_prompts = TargetPromptBatch.from_patchscope_prompts(
            target_patch_prompts
        )
    elif not isinstance(target_patch_prompts, TargetPromptBatch):
        raise ValueError(
            f"patch_prompts must be a TargetPrompt, a list of TargetPrompt or a TargetPromptBatch, got {type(target_patch_prompts)}"
        )
    if len(target_patch_prompts) != len(source_prompts):
        raise ValueError(
            f"Number of prompts ({len(source_prompts)}) does not match number of patch prompts ({len(target_patch_prompts)})"
        )
    hiddens = collect_hidden_states(nn_model, source_prompts, remote=remote)
    # Collect the patch activations for each prompt at each layer
    with nn_model.trace(
        target_patch_prompts.prompts,
        remote=remote,
    ):
        nn_model.model.layers[target_layer].output[0][
            th.arange(len(source_prompts)), target_patch_prompts.index_to_patch
        ] = hiddens
        output = nn_model.lm_head.output
        probs = output[:, -1, :].softmax(-1).cpu().save()
    return probs


def steer(
    nn_model: NNLanguageModel,
    layers: int | list[int],
    steering_vector: th.Tensor,
    factor: float = 1,
    position: int = -1,
    get_module: GetModuleOutput = get_layer_output,
):
    """
    Steer the hidden states of a layer using a steering vector
    Args:
        nn_model: The NNSight model
        layer: The layer to steer
        steering_vector: The steering vector to apply
        factor: The factor to multiply the steering vector by
    """
    if isinstance(layers, int):
        layers = [layers]
    for layer in layers:
        get_module(nn_model, layer)[:, position] += factor * steering_vector


def skip_layers(
    nn_model: NNLanguageModel,
    layer_start: int,
    layer_end: int | None = None,
    num_layers: int | None = None,
    position: int = -1,
):
    """
    Skip the layers AFTER layer_start until layer_end or layer_start + num_layers
    We do this by replacing all the hidden states of the layers to be skipped with the hidden states of the layer at layer_start
    Args:
        nn_model: The NNSight model
        layer_start: The layer to start skipping at
        layer_end: The layer to stop skipping at. If None, skip num_layers layers
        num_layers: The number of layers to skip. If None, skip until layer_end
    """
    if layer_end is None and num_layers is None:
        raise ValueError("Either layer_end or num_layers must be specified")
    elif layer_end is None:
        layer_end = layer_start + num_layers
    elif num_layers is not None:
        raise ValueError("Only one of layer_end or num_layers should be specified")

    for layer in range(layer_start + 1, min(layer_end + 1, get_num_layers(nn_model))):
        get_layer_output(nn_model, layer)[:, position] = get_layer_output(
            nn_model, layer_start
        )[:, position]


@th.no_grad
def patch_attention_lens(
    nn_model: NNLanguageModel,
    source_prompts: list[str] | str,
    target_patch_prompts: TargetPromptBatch | list[TargetPrompt] | TargetPrompt,
    k=5,
    scan=True,
    remote=False,
):
    """
    Replace the hidden state of the patch_prompt.index_to_patch token in the patch_prompt.prompt with the hidden state of the last token of each prompt at each layer.
    Returns the probabilities of the next token in patch_prompt for each prompt for each layer intervention.
    Args:
        nn_model: The NNSight model
        source_prompts: List of prompts or a single prompt to get the hidden states of the last token
        target_patch_prompt: A TargetPrompt object containing the prompt to patch and the index of the token to patch
        k: The number of layers to intervene on
        scan: If looping over this function, set to False after the first call to speed up subsequent calls
        remote: If True, the function will run on the nndif server. See `nnsight.net/status` to check which models are available.
    Returns:
        A tensor of shape (num_prompts, num_layers, vocab_size) containing the probabilities
        of the next token for each prompt at each layer. Tensor is on the CPU.
    """
    if isinstance(source_prompts, str):
        source_prompts = [source_prompts]
    target_patch_prompts = TargetPromptBatch.auto(
        target_patch_prompts, len(source_prompts)
    )
    if len(target_patch_prompts) != len(source_prompts):
        raise ValueError(
            f"Number of prompts ({len(source_prompts)}) does not match number of patch prompts ({len(target_patch_prompts)})"
        )
    hiddens = collect_hidden_states(
        nn_model, source_prompts, remote=remote, get_module_output=get_attention_output
    )
    n_layers = get_num_layers(nn_model)
    probs_l = []
    # Collect the patch activations for each prompt at each layer
    for layer in range(n_layers):
        with nn_model.trace(
            target_patch_prompts.prompts,
            scan=layer == 0 and scan,
            remote=remote,
        ):
            for patch_idx in range(layer, min(layer + k, n_layers)):
                get_attention_output(nn_model, patch_idx)[
                    th.arange(len(source_prompts)), target_patch_prompts.index_to_patch
                ] = hiddens[patch_idx]
            probs_l.append(get_probs(nn_model).cpu().save())
    probs = th.cat(probs_l, dim=0)
    return probs.reshape(n_layers, len(source_prompts), -1).transpose(0, 1)
