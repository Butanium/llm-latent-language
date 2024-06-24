import numpy as np 
from typing import Literal, Union
import pandas as pd

lang_mapping = {
    'ja': 'japanese',
    'de': 'german',
    'en': 'english',
    'nl': 'dutch'
}

harmonize_colors = {
    'gray': 'grey'
}

def get_colors(lang: str):
    colors = {
        'en': ["Red", "Green", "Blue", "Yellow", "Orange", "Purple", "Pink", "Brown", "Black", "White", "Gray", "Cyan", "Magenta", "Lime", "Olive", "Maroon", "Navy", "Teal", "Aqua", "Silver", "Gold"],
        'de': [
                "Rot", "Grün", "Blau", "Gelb", "Orange", "Lila", "Rosa", "Braun", "Schwarz", "Weiß",
                "Grau", "Cyan", "Magenta", "Limonengrün", "Oliv", "Weinrot", "Marineblau", "Türkis", "Aquamarin", "Silber",
                "Gold", "Beige", "Elfenbein", "Koralle", "Indigo", "Lavendel", "Mauve", "Ocker", "Pfirsich", "Perlmutt",
                "Rubinrot", "Smaragdgrün", "Saphirblau", "Schokolade", "Bronze", "Karminrot", "Schiefergrau", "Zinnoberrot", "Königsblau", "Himmelblau"
            ],
        'nl': [
                "Rood", "Groen", "Blauw", "Geel", "Oranje", "Paars", "Roze", "Bruin", "Zwart", "Wit",
                "Grijs", "Cyaan", "Magenta", "Limoengroen", "Olijf", "Bordeauxrood", "Marineblauw", "Turkoois", "Aqua", "Zilver",
                "Goud", "Beige", "Ivoor", "Koraal", "Indigo", "Lavendel", "Mauve", "Oker", "Perzik", "Parelmoer",
                "Robijnrood", "Smaragdgroen", "Saffierblauw", "Chocolade", "Brons", "Karmijnrood", "Leigrijs", "Vermiljoen", "Koningsblauw", "Hemelsblauw"
            ],
        'ja': []
    }
    colors = colors[lang]
    colors = [c.lower() for c in colors]
    return colors 

def get_tastes(lang: str):
    tastes = ["Sweet", "Sour", "Salty", "Bitter", "Umami", "Savory", "Spicy", "Astringent", "Metallic", "Minty", "Hot", "Bland", "Tastes good", "Tastes bad"]
    tastes = [t.lower() for t in tastes]
    return tastes 

def try_eval(x):
    try: 
        return eval(x)
    except:
        return x

def process_word_associations(lang, feat: str, limit_en_concepts: bool = True, min_freq: int = 2, min_feat_freq: int = 2):
    """
    limit_en_concepts: bool -> will restrict to concepts that we have associations for in English
    NOTE: we will always use concept_en -> to avoid collexicfication or multiple words for the same thing... this may be bad
    """
    other_lang = lang
    feat_other = f'feature_{other_lang}_en'
    feat_other_og = f'feature_{other_lang}'
    concept_other = f'concept_{other_lang}' 
    freq_other = f'freq_{other_lang}'
    frac_other = f'frac_{other_lang}'
    df = pd.read_csv(f'/dlabscratch1/veselovs/projects/llm-latent-language/data/norms/augmented_merge_en_{other_lang}.csv')
    df = df.drop_duplicates()
    if other_lang == 'ja':
        df[concept_other] = df['concept_en']
    # fill na
    df[freq_other] = df[freq_other].fillna(0)
    df['freq_en'] = df['freq_en'].fillna(0)
    for col in ['concept_en','feature_en_en', feat_other_og, feat_other]:
        df[col] = df[col].fillna('')
    # we will do a fwd fill
    df[concept_other] = df[concept_other].fillna(method='ffill')
    
    # drop rows
    df = df.dropna(subset=['feature_en_en'])
    df = df[df['feature_en_en'] != '']
    df = df[df[feat_other_og] != '']

    # if color, convert gray -> grey
    if feat == 'color':
        df['feature_en_en'] = df['feature_en_en'].apply(lambda x: harmonize_colors.get(x, x))
        df = df.groupby(['concept_en','feature_en_en', concept_other, feat_other_og, feat_other]).agg('sum').reset_index()
    
    # convert str -> lst 
    df[feat_other] = df[feat_other].apply(try_eval)

    # remove infrequent concepts
    df['total_en'] = df.groupby(['concept_en'])['freq_en'].transform('sum')
    df['total_other'] = df.groupby(['concept_en'])[freq_other].transform('sum')

    df = df[(df['total_en'] >= min_freq) & (df[f'total_other'] >= min_freq)]
    df = df[(df['freq_en'] >= min_feat_freq) | (df[freq_other] >= min_feat_freq)] # one feature has to have more than 
    
    if limit_en_concepts: # see docstring
        temp = df.groupby('concept_en')['freq_en'].sum()
        en_concepts = temp[temp>0].index
        df = df[df['concept_en'].isin(en_concepts)]
    
    # re calculate after we filter
    df['total_en'] = df.groupby(['concept_en'])['freq_en'].transform('sum')
    df['total_other'] = df.groupby(['concept_en'])[freq_other].transform('sum')

    df['frac_en'] = df['freq_en'] / df['total_en']
    df[frac_other] = df[freq_other] / df['total_other']
    
    df['diff'] = df[frac_other] - df['frac_en']
    df = df.sort_values(by='diff', ascending=False)
    
    return df
    
        
    
def sample_concept(df, n_sample: Union[int, str] = 5, mode: Literal['separate', 'combined'] = 'combined') -> tuple:
    """
    Samples concepts from a dataframe for exploration.
    Two modes, select five random from both. 
    Select the union over one. i.e.:
        - english sample: car -> select all German versions of car. 
        - if german Auto -> select all English versions of car
        
    df: pd.DataFrame - augmented dataframe 
    n_sample: int - how many points to sample
    mode: Literal['separate', 'combined'] - 'separate' or 'combined'
    """
    n_sample = len(df) if n_sample == 'all' else n_sample
    col = 'concept_en'
    col_other = [k for k in df.columns if (('concept' in k) and ('en' not in k))][0]
    lang = col_other.split('_')[1]
    col_order = ['concept_en', f'concept_{lang}', 'feature_en_en', 
                 f'feature_{lang}', f'feature_{lang}_en', 'freq_en', f'freq_{lang}',
                 'total_en', 'total_other', 'frac_en', f'frac_{lang}', 'diff']
    concepts, concepts_other = df[col].values, df[col_other].values
    en_concepts = np.random.choice(concepts, n_sample)
    other_concepts = np.random.choice(concepts_other, n_sample)
    
    if mode == 'separate':
        return df[df[col].isin(en_concepts)].sort_values(by=[col, 'diff']), \
                df[df[col_other].isin(other_concepts)].sort_values(by=[col_other, 'diff'])[col_order]
    
    elif mode == 'combined':
        lst_en = en_concepts.tolist()
        lst_other = other_concepts.tolist()
        for en_c in en_concepts:
            lst_other.extend(df[df[col] == en_c][col_other].unique().tolist())
        for en_o in other_concepts:
            lst_en.extend(df[df[col_other] == en_o][col].unique().tolist()) 
        
        return df[(df[col].isin(lst_en)) & (df[col_other].isin(lst_other))]\
            .sort_values(by=['diff', col, col_other], ascending=False)[col_order], None
    
    else:
        raise ValueError(f'mode must be `separate`or `combined` currently - {mode}')
