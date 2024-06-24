import pandas as pd 
from attr import define, field 
import sys
import os 

sys.path.append('generalization/')
sys.path.append('src/')

from helpers import get_colors, get_tastes
from utils import load_yaml
from generate_dfs import generate_bn_dataset
from translation_tools import get_bn_synsets, filter_synsets
import babelnet as bn

feat_mapping = {
    'colors': get_colors
}


@define 
class AbstractLang:
    lang: str = field()
    data: pd.DataFrame = field(default=None)
    feats: list = field(default=['colors'])
    feat_fns: list = field(default=None)
    
    # paths 
    data_dir: str = field(default=None)
    path: str = field(default=None)
    config_path: str = field(default='src/config.yaml')
    save_path: str = field(default=None)
    data_kwargs: dict = field(factory=dict)
    
    def __attrs_post_init__(self):
        self._add_init()
        if self.data is None: 
            config = load_yaml(self.config_path)
            self.data_dir = config['norms_dir']
            self.path = config['norms_paths'][self.lang] 
            self.path = os.path.join(self.data_dir, self.path)
            self.data = load_arbitrary_path(self.path, **self.data_kwargs)
            
        # if len(self.feats) != self.feat_functions:
        #     raise ValueError(f'Ensure self.feats has same length as self.feat_fns ({len(self.feats)} != {len(self.feat_fns)})')
        
        self.save_path = os.path.join("/".join(self.path.split('/')[:-1]), 'augmented.csv')
        self.feat_fns = [feat_mapping[feat] for feat in self.feats]
        
    def translate_cols(self, df, translate_cols: list = ['concept', 'feature']):
        for col in translate_cols:
            df[col + '_en'] = self.translate_list_to_en(df[col + '_og'].to_list())
        return df 
        
    def translate_list_to_en(self, translate_list: list):
        trans = generate_bn_dataset(self.lang, 'en', translate_list, prune_empty=False, keep_original_word=True)['en'].values
        return trans
        
    def _add_init(self):
        pass
        

def load_arbitrary_path(path, sep=None, **kwargs):
    if '.xlsx' in path:
        return pd.read_excel(path, **kwargs)
    elif '.csv' in path:
        return pd.read_csv(path, **kwargs)
    elif ('.txt' in path) & (sep is None):
        raise ValueError('For .txt format, must indicate seperator (sep)')
    elif ('.txt' in path) & (sep is not None):
        return pd.read_csv(path, sep=sep, **kwargs)

    