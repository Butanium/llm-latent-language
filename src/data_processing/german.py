from attr import define, field 
import pandas as pd 
import sys
import os

sys.path.append('generalization/')
sys.path.append('src/')

from abstract_lang import AbstractLang, feat_mapping

@define 
class GermanData(AbstractLang):
    lang: str = field(default='de')
    
    def prepare_lang(self):
        df = pd.DataFrame()
        for f, fn in zip(self.feats, self.feat_fns):
            fn = fn(self.lang)
            data = self.data[self.data['dirty'].str.contains('|'.join(fn))]
            data['feature_og'] = data['dirty'].apply(lambda x: self.check_which_color(x, fn))
            data = data.explode('feature_og')
            data = data.groupby(['concept_og','feature_og'])['freq'].sum().reset_index()
            
            data['feature'] = f
            df = pd.concat([df, data], axis=0)
        
        df = self.translate_cols(df)

        df.to_csv(self.save_path, index=False)
        return df 
    
    def _add_init(self):
        self.data_kwargs = {"sep": '\t', "header":None, "names": ['concept_og', 'dirty', 'freq']}
        
    @staticmethod
    def check_which_color(x, colors):
        found_colors = []
        for c in colors:
            if c in x:
                found_colors.append(c)
        return found_colors
    
if __name__=='__main__':
    ed = GermanData()
    ed.prepare_lang()