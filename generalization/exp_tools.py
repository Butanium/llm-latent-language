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
from utils import ulist, get_tokenizer
from typing import Callable
from warnings import warn
from typing import Optional, Union
import re
from interventions import *
from typing import Literal

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


def load_lang(lang):
    path = DATA_PATH / "langs" / lang / "clean.csv"
    return pd.read_csv(path)


def token_prefixes(token_str: str):
    n = len(token_str)
    tokens = [token_str[:i] for i in range(1, n + 1)]
    return tokens


SPACE_TOKENS = ["▁", "Ġ", " "]


def add_spaces(tokens):
    return sum([[s + token for token in tokens] for s in SPACE_TOKENS], []) + tokens


# TODO?: Add capitalization


def byte_string_to_list(input_string):
    # Find all parts of the string: either substrings or hex codes
    parts = re.split(r"(\\x[0-9a-fA-F]{2})", input_string)
    result = []
    for part in parts:
        if re.match(r"\\x[0-9a-fA-F]{2}", part):
            # Convert hex code to integer
            result.append(int(part[2:], 16))
        else:
            if part:  # Skip empty strings
                result.append(part)
    return result


def unicode_prefixes(tok_str):
    encoded = str(tok_str.encode())[2:-1]
    if "\\x" not in encoded:
        return []  # No bytes in the string
    chr_list = byte_string_to_list(encoded)
    if isinstance(chr_list[0], int):
        first_byte_token = [
            f"<{hex(chr_list[0]).upper()}>"
        ]  # For llama2 like tokenizer, this is how bytes are represented
    else:
        first_byte_token = []
    # We need to convert back to latin1 to get the character
    for i, b in enumerate(chr_list):
        # those bytes are not valid latin1 characters and are shifted by 162 in Llama3 and Qwen
        if isinstance(b, str):
            continue
        if b >= 127 and b <= 160:
            chr_list[i] += 162
        chr_list[i] = chr(chr_list[i])
    # Convert back to string
    vocab_str = "".join(
        chr_list
    )  # This is the string that will be in the tokenizer vocab for Qwen and Llama3
    return first_byte_token + token_prefixes(vocab_str)


def process_tokens(words: str | list[str], tok_vocab):
    if isinstance(words, str):
        words = [words]
    final_tokens = []
    for word in words:
        with_prefixes = token_prefixes(word) + unicode_prefixes(word)
        with_spaces = add_spaces(with_prefixes)
        for word in with_spaces:
            if word in tok_vocab:
                final_tokens.append(tok_vocab[word])
    return ulist(final_tokens)


def process_tokens_with_tokenization(words: str | list[str], tokenizer):
    if isinstance(words, str):
        words = [words]
    final_tokens = []
    for word in words:
        # If you get the value error even with add_prefix_space=False,
        # you can use the following hacky code to get the token without the prefix
        # hacky_token = tokenizer("🍐", add_special_tokens=False).input_ids
        # length = len(hacky_token)
        # tokens = tokenizer("🍐" + word, add_special_tokens=False).input_ids
        # assert (
        #     tokens[:length] == hacky_token
        # ), "I didn't expect this to happen, please check this code"
        # if len(tokens) > length:
        #     final_tokens.append(tokens[length])

        token = tokenizer(word, add_special_tokens=False).input_ids[0]
        token_with_start_of_word = tokenizer(
            " " + word, add_special_tokens=False
        ).input_ids[0]
        if token == token_with_start_of_word:
            raise ValueError(
                "Seems like you use a tokenizer that wasn't initialized with add_prefix_space=False. Not good :("
            )
        final_tokens.append(token)
        if (
            token_with_start_of_word
            != tokenizer(" ", add_special_tokens=False).input_ids[0]
        ):
            final_tokens.append(token_with_start_of_word)
    return ulist(final_tokens)


def get_mean_activations(nn_model, prompts_str, batch_size=32):
    dataloader = DataLoader(prompts_str, batch_size=batch_size, shuffle=False)
    acts = []
    for batch in tqdm(dataloader):
        acts.append(collect_activations(nn_model, batch))
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


# TODO: merge with logit_lens
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


def _next_token_probs_unsqueeze(nn_model, prompt, scan=True):
    probs = next_token_probs(nn_model, prompt)
    return probs.unsqueeze(1)  # Add a fake layer dimension


method_to_fn = {
    "next_token_probs": _next_token_probs_unsqueeze,
    "logit_lens": logit_lens,
    "logit_lens_llama": logit_lens_llama,
    "patchscope_lens": patchscope_lens,
}


@dataclass
class Prompt:
    prompt: str
    target_tokens: list[int]
    latent_tokens: dict[str, list[int]]
    target_strings: str
    latent_strings: dict[str, str | list[str]]

    @classmethod
    def from_strings(cls, prompt, target_strings, latent_strings, tokenizer):
        tokenizer = get_tokenizer(tokenizer)
        target_tokens = process_tokens_with_tokenization(target_strings, tokenizer)
        latent_tokens = {
            lang: process_tokens_with_tokenization(words, tokenizer)
            for lang, words in latent_strings.items()
        }
        return cls(
            target_tokens=target_tokens,
            latent_tokens=latent_tokens,
            target_strings=target_strings,
            latent_strings=latent_strings,
            prompt=prompt,
        )

    def get_target_probs(self, probs):
        target_probs = probs[:, :, self.target_tokens].sum(dim=2)
        return target_probs.cpu()

    def get_latent_probs(self, probs, layer=None):
        latent_probs = {
            lang: probs[:, :, tokens].sum(dim=2).cpu()
            for lang, tokens in self.latent_tokens.items()
        }
        if layer is not None:
            latent_probs = {lang: probs_[:, layer] for lang, probs_ in latent_probs.items()}
        return latent_probs

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
        return self.get_target_probs(probs), self.get_latent_probs(probs)

    def has_no_collisions(self, ignore_langs: Optional[str | list[str]] = None):
        tokens = self.target_tokens[:]  # Copy the list
        if isinstance(ignore_langs, str):
            ignore_langs = [ignore_langs]
        if ignore_langs is None:
            ignore_langs = []
        for lang, lang_tokens in self.latent_tokens.items():
            if lang in ignore_langs:
                continue
            tokens += lang_tokens
        return len(tokens) == len(set(tokens))


@th.no_grad
def run_prompts(
    nn_model,
    prompts,
    batch_size=32,
    method: str | Callable = "logit_lens",
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
    if isinstance(method, str):
        get_probs = method_to_fn[method]
    else:
        get_probs = method
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


def filter_prompts(prompts, ignore_langs: Optional[str | list[str]] = None):
    return [prompt for prompt in prompts if prompt.has_no_collisions(ignore_langs)]
