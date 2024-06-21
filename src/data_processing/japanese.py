from attr import define, field 
import pandas as pd 
import sys
import os

sys.path.append('generalization/')
sys.path.append('src/')

from abstract_lang import AbstractLang, feat_mapping

@define 
class JapaneseData(AbstractLang):
    lang: str = field(default='ja')
    
    feature_mapping: dict = field(default={'colors': 'visual colour'})
    
    def prepare_lang(self):
        df = pd.DataFrame()
        for f, fn in zip(self.feats, self.feat_fns):
            fn = fn(self.lang)
            data = self.data[self.data['BR_Label'] == self.feature_mapping[f]].copy()
            data = data.rename(columns={'Concept': 'concept_en', 'Feature': 'feature_og', 'Prod_Freq': 'freq'}).drop('BR_Label', axis=1)
            data['feature'] = f
            df = pd.concat([df, data], axis=0)
        
        df = self.translate_cols(df, translate_cols=['feature'])

        df.to_csv(self.save_path, index=False)
        return df 
        
    @staticmethod
    def check_which_color(x, colors):
        found_colors = []
        for c in colors:
            if c in x:
                found_colors.append(c)
        return found_colors
    
if __name__=='__main__':
    ed = JapaneseData()
    ed.prepare_lang()