from __future__ import annotations
from dataclasses import dataclass
import torch as th
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
from nnsight.models.UnifiedTransformer import UnifiedTransformer

DATA_PATH = Path(__file__).resolve().parent.parent / "data"


def load_lang(lang):
    path = DATA_PATH / "langs" / lang / "clean.csv"
    return pd.read_csv(path)


def token_prefixes(token_str: str):
    n = len(token_str)
    tokens = [token_str[:i] for i in range(1, n + 1)]
    return tokens


def add_spaces(tokens):
    return ["▁" + t for t in tokens] + tokens


def capitalizations(tokens):
    return list(set(tokens))


def unicode_prefix_tokid(zh_char, tokenizer):
    start = zh_char.encode().__str__()[2:-1].split("\\x")[1]
    unicode_format = "<0x%s>"
    start_key = unicode_format % start.upper()
    if start_key in tokenizer.get_vocab():
        return tokenizer.get_vocab()[start_key]
    return None


def process_tokens(words: str | list[str], tokenizer, lang=None):
    if isinstance(words, str):
        words = [words]
    final_tokens = []
    for word in words:
        with_prefixes = token_prefixes(word)
        with_spaces = add_spaces(with_prefixes)
        with_capitalizations = capitalizations(with_spaces)
        for cap_word in with_capitalizations:
            if cap_word in tokenizer.get_vocab():
                final_tokens.append(tokenizer.get_vocab()[cap_word])
        if lang in ["zh", "ru"]:
            tokid = unicode_prefix_tokid(word, tokenizer)
            if tokid is not None:
                final_tokens.append(tokid)
    return list(set(final_tokens))


@th.no_grad
def logit_lens(nn_model: UnifiedTransformer, prompts, scan=True):
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
    with nn_model.trace(prompts, scan=scan) as tracer:
        hiddens_l = [
            layer.output[
                th.arange(len(tok_prompts.input_ids), device=layer.output.device),
                last_token_index.to(layer.output.device),
            ].unsqueeze(1)
            for layer in nn_model.blocks
        ]
        hiddens = th.cat(hiddens_l, dim=1)
        ln_out = nn_model.ln_final(hiddens)
        logits = nn_model.unembed(ln_out).cpu().save()
        output = nn_model.unembed.output.cpu().save()
        probs = logits.softmax(-1).cpu().save()
    if not th.allclose(
        logits[:, -1, :],
        output[
            th.arange(len(tok_prompts.input_ids), device=logits.device),
            last_token_index.to(logits.device),
        ],
        atol=1e-4,
    ):
        diff = (
            (
                logits[:, -1, :]
                - output[
                    th.arange(len(tok_prompts.input_ids), device=logits.device),
                    last_token_index.to(logits.device),
                ]
            )
            .abs()
            .max()
        )
        raise RuntimeError(f"Logits and output don't match. Max diff: {diff}")
    return probs


@dataclass
class Prompt:
    prompt: str
    target_tokens: list[int]
    latent_tokens: dict[str, list[int]]
    target_string: str
    latent_strings: dict[str, str | list[str]]

    @th.no_grad
    def run(self, nn_model):
        """
        Run the prompt through the model and return the probabilities of the next token for both the target and latent languages.
        """
        probs = logit_lens(nn_model, self.prompt)
        target_probs = probs[:, :, self.target_tokens].sum(dim=2)
        latent_probs = {
            lang: probs[:, :, tokens].sum(dim=2)
            for lang, tokens in self.latent_tokens.items()
        }
        return target_probs.cpu(), latent_probs.cpu()


@th.no_grad
def run_prompts(nn_model, prompts, batch_size=32):
    """
    Run a list of prompts through the model and return the probabilities of the next token for both the target and latent languages.

    Returns:
        Two tensors target_probs and latent_probs of shape (num_prompts, num_layers)
    """
    # Todo: split in batches
    target_probs = []
    latent_probs = {lang: [] for lang in prompts[0].latent_tokens.keys()}
    str_prompts = [prompt.prompt for prompt in prompts]
    dataloader = DataLoader(str_prompts, batch_size=batch_size)
    probs = []
    scan = True
    for prompt_batch in tqdm(dataloader, total=len(dataloader)):
        probs.append(logit_lens(nn_model, prompt_batch, scan=scan))
        scan = False
    probs = th.cat(probs)
    for i, prompt in enumerate(prompts):
        target_probs.append(probs[i, :, prompt.target_tokens].sum(dim=1))
        for lang, tokens in prompt.latent_tokens.items():
            latent_probs[lang].append(probs[i, :, tokens].sum(dim=1))
    target_probs = th.stack(target_probs).cpu()
    latent_probs = {lang: th.stack(probs).cpu() for lang, probs in latent_probs.items()}
    return target_probs, latent_probs
