from attr import define, field 
import pandas as pd 
import sys
import os

sys.path.append('generalization/')
sys.path.append('src/')

from abstract_lang import AbstractLang, feat_mapping

@define 
class EnglishData(AbstractLang):
    auxilary_path: str = field(default='data/norms/japanese/EnglishNorms.csv')
    
    feature_mapping: dict = field(default={'colors': 'visual colour'})

    
    def prepare_lang(self):
        df = pd.DataFrame()
        for f, fn in zip(self.feats, self.feat_fns):
            fn = fn(self.lang)
            data = self.data[self.data['translated'].isin(fn)]\
                    .groupby(['cue', 'translated'])[['frequency_translated']].sum().reset_index()
            data['feature'] = f
            df = pd.concat([df, data], axis=0)
        
        df = df.rename(columns={'cue': 'concept_en','translated': 'feature_en', 'frequency_translated': 'freq'})
        df = pd.concat([df, self.auxilary_dataset()])
        df = df.groupby(['concept_en', 'feature_en', 'feature'])['freq'].sum().reset_index()
        
        df.to_csv(self.save_path, index=False)
        return df 
    
    def auxilary_dataset(self):
        # TODO: add THE ENGLISH NORMS FROM JAPANESE DATASET
        aux = pd.read_csv(self.auxilary_path)
        df = pd.DataFrame()
        for f, fn in zip(self.feats, self.feat_fns):
            fn = fn(self.lang)
            data = aux[aux['BR_Label'] == self.feature_mapping[f]].copy()
            data = data.rename(columns={'Concept': 'concept_en', 'Features': 'feature_en', 'Prod_Freq': 'freq'}).drop('BR_Label', axis=1)
            data['feature'] = f
            df = pd.concat([df, data], axis=0)

        return df 
        
    
if __name__=='__main__':
    ed = EnglishData('en')
    ed.prepare_lang()