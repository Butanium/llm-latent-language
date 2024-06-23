import pandas as pd 
import numpy as np 
import os 
import pickle

import sys

sys.path.append('./generalization')

from translation_tools import generate_bn_dataset

data_path = 'data/norms/'
langs = ['EnglishNorms.csv', 'JapaneseNorms.csv']

file_info = {
    'JapaneseNorms.csv':
        {'feature_lang': 'ja'}
}

feats = ['visual colour', 'taste']
save_dir = 'data/norms'

def prepare_list_of_clayman():
    for lang in langs: 
        file_name = lang.split('.')[0]
        if 'English' in lang:
            continue 
        path = os.path.join(data_path, lang)
        
        df = pd.read_csv(path)
        df = df[df['BR_Label'].isin(feats)]
        en_words = generate_bn_dataset(file_info[lang]['feature_lang'], 'en', df['Feature'].to_list(), prune_empty=False)['en'].values
        print(en_words)
        df['en_feature'] = en_words
        
        df.to_csv(os.path.join(data_path, file_name+'_augmented.csv'), index=False)
        
            
if __name__=='__main__':
    prepare_list_of_clayman()
        
            