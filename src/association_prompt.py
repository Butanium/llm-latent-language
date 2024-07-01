import os
import pandas as pd
from attr import define, field

from helpers import get_colors
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data"


@define
class LoadAssociations:
    lang: str = field()
    full_lang: str = field(default=None)
    full_feat: str = field(default=None)
    df: pd.DataFrame = field(default=None)

    data_dir: str = field(default=DATA_PATH / "norms/processed/")
    data_path: str = field(default=None)

    en_colors: dict = field(default=None)
    other_colors: dict = field(default=None)

    feat: str = field(default="color")

    lang_colors: dict = field(default=None)
    langs: dict = field(default=None)

    prompt_template: str = field(
        default='"{lang}": "{obj}" - "{color_lang}": "{color}"'
    )

    def __attrs_post_init__(self):
        if self.feat != "color":
            raise NotImplementedError("Only supports feature == color")
        self.data_path = self.data_dir / (self.lang + ".csv")
        self.df = pd.read_csv(self.data_path)

        self.lang_colors = {"en": "color", "de": "Farbe", "ja": "色", "nl": "kleur"}
        self.langs = {
            "en": "english",
            "de": "Deutsche",
            "nl": "Nederlands",
            "ja": "日本語",
        }

        self.full_lang = self.langs[self.lang]
        self.full_feat = self.lang_colors[self.lang]

        self._process_df()

        self.en_colors = get_colors("en")
        self.other_colors = get_colors(self.lang)

    def _process_df(self):
        self.df = self.df.dropna(
            subset=[f"concept_{self.lang}", f"feature_{self.lang}"]
        )
        #
        self.df = (
            self.df.sort_values(by=f"freq_{self.lang}", ascending=False)
            .groupby(f"concept_{self.lang}")
            .head(1)
        )

    def generate_prompt(self, predict=None, n_examples=4):
        """
        return a few shot language extraction prompt

        "obj1"  -  color^lang: "color"
        """
        if predict is None:
            rows = self.df.sample(n_examples + 1)
        else:
            df = self.df[
                self.df[f"concept_{self.lang}"] != predict[f"concept_{self.lang}"]
            ]
            rows = df.sample(
                n_examples + 1
            )  # last example doesn't count, but doing it so we iterate over the row

        prompt = ""
        for i, (_, row) in enumerate(rows.iterrows()):
            if (i == n_examples) and (predict is None):
                temp_prompt = self.prompt_template.format(
                    lang=self.full_lang,
                    obj=row[f"concept_{self.lang}"],
                    color_lang=self.full_feat,
                    color="",
                )[:-1]
                prompt += temp_prompt
            elif (i == n_examples) and (predict is not None):
                temp_prompt = self.prompt_template.format(
                    lang=self.full_lang,
                    obj=predict[f"concept_{self.lang}"],
                    color_lang=self.full_feat,
                    color="",
                )[:-1]
                prompt += temp_prompt
            else:
                temp_prompt = self.prompt_template.format(
                    lang=self.full_lang,
                    obj=row[f"concept_{self.lang}"],
                    color_lang=self.full_feat,
                    color=row[f"feature_{self.lang}"],
                )
                prompt += temp_prompt + "\n"
        return (prompt, row[f"concept_{self.lang}"])

    def generate_all_prompts(self, n_examples: int = 4):
        all_prompts = []
        for _, row in self.df.iterrows():
            all_prompts.append(self.generate_prompt(row, n_examples))
        return all_prompts


if __name__ == "__main__":
    la = LoadAssociations("de")
    resp = la.generate_all_prompts()
    print(resp[0])
