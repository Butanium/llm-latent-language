from nnsight.models.UnifiedTransformer import UnifiedTransformer
from nnsight.models.LanguageModel import LanguageModelProxy, LanguageModel
from nnsight.envoy import Envoy
import torch as th
from typing import Union, Callable

NNLanguageModel = Union[UnifiedTransformer, LanguageModel]
GetModuleOutput = Callable[[NNLanguageModel, int], LanguageModelProxy]


def get_num_layers(nn_model: NNLanguageModel):
    """
    Get the number of layers in the model
    Args:
        nn_model: The NNSight model
    Returns:
        The number of layers in the model
    """
    if isinstance(nn_model, UnifiedTransformer):
        return len(nn_model.blocks)
    else:
        return len(nn_model.model.layers)


def get_layer(nn_model: NNLanguageModel, layer: int) -> Envoy:
    """
    Get the layer of the model
    Args:
        nn_model: The NNSight model
        layer: The layer to get
    Returns:
        The Envoy for the layer
    """
    if isinstance(nn_model, UnifiedTransformer):
        return nn_model.blocks[layer]
    else:
        return nn_model.model.layers[layer]


def get_layer_input(nn_model: NNLanguageModel, layer: int) -> LanguageModelProxy:
    """
    Get the hidden state input of a layer
    Args:
        nn_model: The NNSight model
        layer: The layer to get the input of
    Returns:
        The Proxy for the input of the layer
    """
    return get_layer(nn_model, layer).input[0][0]


def get_layer_output(nn_model: NNLanguageModel, layer: int) -> LanguageModelProxy:
    """
    Get the output of a layer
    Args:
        nn_model: The NNSight model
        layer: The layer to get the output of
    Returns:
        The Proxy for the output of the layer
    """
    output = get_layer(nn_model, layer).output
    if isinstance(nn_model, UnifiedTransformer):
        return output
    else:
        return output[0]


def get_attention(nn_model: NNLanguageModel, layer: int) -> Envoy:
    """
    Get the attention module of a layer
    Args:
        nn_model: The NNSight model
        layer: The layer to get the attention module of
    Returns:
        The Envoy for the attention module of the layer
    """
    if isinstance(nn_model, UnifiedTransformer):
        return nn_model.blocks[layer].attn
    else:
        return nn_model.model.layers[layer].self_attn


def get_attention_output(nn_model: NNLanguageModel, layer: int) -> LanguageModelProxy:
    """
    Get the output of the attention block of a layer
    Args:
        nn_model: The NNSight model
        layer: The layer to get the output of
    Returns:
        The Proxy for the output of the attention block of the layer
    """
    output = get_attention(nn_model, layer).output
    if isinstance(nn_model, UnifiedTransformer):
        return output
    else:
        return output[0]


def get_logits(nn_model: NNLanguageModel) -> LanguageModelProxy:
    """
    Get the logits of the model
    Args:
        nn_model: The NNSight model
    Returns:
        The Proxy for the logits of the model
    """
    if isinstance(nn_model, UnifiedTransformer):
        return nn_model.unembed.output
    else:
        return nn_model.lm_head.output


def get_next_token_probs(nn_model: NNLanguageModel) -> LanguageModelProxy:
    """
    Get the probabilities of the model
    Args:
        nn_model: The NNSight model
    Returns:
        The Proxy for the probabilities of the model
    """
    return get_logits(nn_model)[:, -1, :].softmax(-1)


@th.no_grad
def collect_activations(
    nn_model: NNLanguageModel,
    prompts,
    layers=None,
    get_activations: GetModuleOutput = get_layer_output,
    remote=False,
):
    tok_prompts = nn_model.tokenizer(prompts, return_tensors="pt", padding=True)
    # Todo?: This is a hacky way to get the last token index but it works for both left and right padding
    last_token_index = (
        tok_prompts.attention_mask.flip(1).cumsum(1).bool().int().sum(1).sub(1)
    )
    if layers is None:
        layers = range(get_num_layers(nn_model))
    # Collect the hidden states of the last token of each prompt at each layer
    with nn_model.trace(prompts, remote=remote):
        hiddens = [
            get_activations(nn_model, layer)[
                th.arange(len(tok_prompts.input_ids)),
                last_token_index,
            ]
            .cpu()
            .save()
            for layer in layers
        ]
    return hiddens
