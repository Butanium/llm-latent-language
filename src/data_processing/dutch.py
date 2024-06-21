from attr import define, field 
import pandas as pd 
import sys
import os

sys.path.append('generalization/')
sys.path.append('src/')

from abstract_lang import AbstractLang, feat_mapping

@define 
class DutchData(AbstractLang):
    
    def prepare_lang(self):
        df = pd.DataFrame()
        for f, fn in zip(self.feats, self.feat_fns):
            fn = fn(self.lang)
            data = self.data.melt(value_vars=['asso1', 'asso2', 'asso3'], id_vars=['exemplar','participant'])\
                     .drop(columns=['variable'])
            data = data[data['value'].str.lower() != 'x']
            data = data.groupby(['exemplar','value'])['participant'].nunique().reset_index()
            data = data[data['value'].str.lower().isin(fn)]
            data = data.rename(columns={'exemplar': 'concept_og', 'value': 'feature_og', 'participant': 'freq'})
            
            data['feature'] = f
            df = pd.concat([df, data], axis=0)
        
        df = self.translate_cols(df)
        
        df.to_csv(self.save_path, index=False)
        return df 
    

    
if __name__=='__main__':
    ed = DutchData('nl')
    ed.prepare_lang()