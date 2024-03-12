from dataclasses import dataclass
import torch as th
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
from random import sample
from tqdm.auto import tqdm

__all__ = ["process_tokens"]


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


def process_tokens(token_str: str, tokenizer, lang=None):
    with_prefixes = token_prefixes(token_str)
    with_spaces = add_spaces(with_prefixes)
    with_capitalizations = capitalizations(with_spaces)
    final_tokens = []
    for tok in with_capitalizations:
        if tok in tokenizer.get_vocab():
            final_tokens.append(tokenizer.get_vocab()[tok])
    if lang in ["zh", "ru"]:
        tokid = unicode_prefix_tokid(token_str, tokenizer)
        if tokid is not None:
            final_tokens.append(tokid)
    return final_tokens


@th.no_grad
def logit_lens(nn_model, prompts):
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
    # Todo?: This is a hacky way to get the last token index
    last_token_index = (
        tok_prompts.attention_mask.flip(1).cumsum(1).bool().int().sum(1).sub(1)
    )
    with nn_model.trace(prompts) as tracer:
        hiddens_l = [
            layer.output[0][
                th.arange(len(tok_prompts.input_ids), device=layer.output[0].device),
                last_token_index.to(layer.output[0].device),
            ].unsqueeze(1)
            for layer in nn_model.model.layers
        ]
        hiddens = th.cat(hiddens_l, dim=1)
        rms_out = nn_model.model.norm(hiddens)
        logits = nn_model.lm_head(rms_out).cpu().save()
        output = nn_model.lm_head.output.cpu().save()
        probs = logits.softmax(-1).cpu().save()
    assert th.allclose(
        logits[:, -1, :],
        output[
            th.arange(len(tok_prompts.input_ids), device=logits.device),
            last_token_index.to(logits.device),
        ],
    )
    return probs


def load_lang(lang):
    path = Path("data") / "langs" / lang / "clean.csv"
    return pd.read_csv(path)


def get_translations(
    langs, tokenizer=None, multi_token_only=False, single_token_only=False
):
    """
    Load translations from multiple languages and filter by token type if necessary
    """
    assert not (
        multi_token_only and single_token_only
    ), "Cannot have both multi_token_only and single_token_only"
    assert tokenizer is not None or not (
        multi_token_only or single_token_only
    ), "Cannot filter tokens without a tokenizer"
    dfs = [load_lang(lang) for lang in langs]
    dfs_dict = {f"{lang}": df for lang, df in zip(langs, dfs)}
    merged_df = pd.DataFrame(
        {
            name: df.set_index("word_original")["word_translation"]
            for name, df in dfs_dict.items()
        }
    )
    if "en" not in langs:
        merged_df = (
            merged_df.dropna(how="any")
            .reset_index()
            .rename(columns={"word_original": "en"})
        )
    else:
        merged_df = merged_df.dropna(how="any").reset_index(drop=True)
    print(f"Found {len(merged_df)} translations")
    if not multi_token_only and not single_token_only:
        return merged_df
    for idx, row in merged_df.iterrows():
        for lang in langs:
            if (
                row[lang] in tokenizer.get_vocab()
                or "▁" + row[lang] in tokenizer.get_vocab()
            ):
                if multi_token_only:
                    merged_df.drop(idx, inplace=True)
                    break
            elif single_token_only:
                merged_df.drop(idx, inplace=True)
                break
    print(f"Filtered to {len(merged_df)} translations")
    return merged_df


lang2name = {
    "fr": "Français",
    "de": "Deutsch",
    "ru": "Русский",
    "en": "English",
    "zh": "中文",
}


@dataclass
class Prompt:
    prompt: str
    target_tokens: list[int]
    latent_tokens: list[int]
    target_string: str
    latent_string: str

    @th.no_grad
    def run(self, nn_model):
        """
        Run the prompt through the model and return the probabilities of the next token for both the target and latent languages.
        """
        probs = logit_lens(nn_model, self.prompt)
        target_probs = probs[:, :, self.target_tokens].sum(dim=2)
        latent_probs = probs[:, :, self.latent_tokens].sum(dim=2)
        return target_probs, latent_probs


@th.no_grad
def run_prompts(nn_model, prompts, batch_size=32):
    """
    Run a list of prompts through the model and return the probabilities of the next token for both the target and latent languages.

    Returns:
        Two tensors target_probs and latent_probs of shape (num_prompts, num_layers)
    """
    # Todo: split in batches
    target_probs = []
    latent_probs = []
    str_prompts = [prompt.prompt for prompt in prompts]
    dataloader = DataLoader(str_prompts, batch_size=batch_size)
    probs = []
    for prompt_batch in tqdm(dataloader):
        probs.append(logit_lens(nn_model, prompt_batch))
    probs = th.cat(probs)
    for i, prompt in enumerate(prompts):
        target_probs.append(probs[i, :, prompt.target_tokens].sum(dim=1))
        latent_probs.append(probs[i, :, prompt.latent_tokens].sum(dim=1))
    target_probs = th.stack(target_probs)
    latent_probs = th.stack(latent_probs)
    return target_probs, latent_probs


def translation_prompts(df, tokenizer, input_lang, target_lang, latent_lang, n=5):
    """
    Get a translation prompt from input_lang to target_lang for each row in the dataframe.

    Args:
        df: DataFrame containing translations
        tokenizer: Huggingface tokenizer
        input_lang: Language to translate from
        target_lang: Language to translate to
        n: Number of few-shot examples for each translation

    Returns:
        List of Prompt objects
    """
    assert (
        len(df) > n
    ), f"Not enough translations from {input_lang} to {target_lang} for n={n}"
    prompts = []
    for idx, row in tqdm(df.iterrows()):
        idxs = df.index.tolist()
        idxs.remove(idx)
        fs_idxs = sample(idxs, n)
        prompt = ""
        for fs_idx in fs_idxs:
            fs_row = df.loc[fs_idx]
            prompt += f'{lang2name[input_lang]}: "{fs_row[input_lang]}" - {lang2name[target_lang]}: "{fs_row[target_lang]}"\n'
        prompt += f'{lang2name[input_lang]}: "{row[input_lang]}" - {lang2name[target_lang]}: "'
        target_tokens = process_tokens(row[target_lang], tokenizer, lang=target_lang)
        latent_tokens = process_tokens(row[latent_lang], tokenizer, lang=latent_lang)
        prompts.append(
            Prompt(
                prompt, target_tokens, latent_tokens, row[target_lang], row[latent_lang]
            )
        )
    return prompts
