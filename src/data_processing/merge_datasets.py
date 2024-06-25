"""
Take in a list of langs 
Merge on the english columns (concepts)
"""
from attr import define, field 
import pandas as pd 
import sys 
import os 

sys.path.append('generalization/')
sys.path.append('src/')

from utils import load_yaml
from helpers import lang_mapping


@define 
class MergeProtocol:
    anchor_lang: str = field(default='en')
    langs: list = field(default=['ja', 'en', 'de', 'nl'])
    feat: str = field(default='color')
    
    config: dict = field(default=None)
    config_path: str = field(default='src/config.yaml')
    data_dir: str = field(default=None)
    
    def __attrs_post_init__(self):
        if self.config is None: 
            self.config = load_yaml(self.config_path)
            
        self.data_dir = self.config['norms_dir']
    
    def load_file(self, lang):
        df = pd.read_csv(os.path.join(self.data_dir, lang_mapping[lang], 'augmented.csv'))
        try:
            df['concept_en'] = df['concept_en'].apply(eval)
            df = df.explode(['concept_en'])
        except Exception as e:
            print(f'concept_en not a list for {lang}')
        
        df = df.rename(columns={'feature_en': f'feature_{lang}_en',
                                'feature_og': f'feature_{lang}', 
                                'freq': f'freq_{lang}', 'concept_og': f'concept_{lang}'})
        df = df.drop(columns=['feature'])
        return df 
    
    def _return_en_concept(self, en_features, other_feats):
        feats = [None]
        for f in en_features: 
            temp_f = f"'{f}'"
            if temp_f in other_feats:
                feats.append(f)
        if len(feats) > 1:
            feats = feats[1:]
        return feats
            
    def merge(self):
        en = self.load_file(self.anchor_lang)
        en_features = en['feature_en_en'].values
        merges = {}
        for lang in self.langs:
            if lang == self.anchor_lang:
                continue 
            df = self.load_file(lang)
            df['en_mapped'] = df[f'feature_{lang}_en'].apply(lambda x: self._return_en_concept(en_features, x))
            df = df.explode(['en_mapped']).rename(columns={'en_mapped': 'feature_en_en'})
            lang_merge = en.merge(df, on=['concept_en', 'feature_en_en'], how='outer')
            merges[lang] = lang_merge
            lang_merge.to_csv(os.path.join(self.data_dir, 
                                           f'augmented_merge_{self.anchor_lang}_{lang}.csv'), index=False)
        return merges
        
if __name__=='__main__':
    mp = MergeProtocol()
    mp.merge()