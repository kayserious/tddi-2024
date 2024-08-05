adjectives = ['diğer','bu','birçok','başka','farklı']
nouns = ['firma','marka','şirket','operatör','banka','mağaza','platform','şebeke']


import pandas as pd
import itertools
from coreference_resolver import coreference_resolver


def multicheck(value,array):
    logics = []
    for arrayitem in array:
        if arrayitem in value:
            logics.append(True)
        else:
            logics.append(False)
    return any(logics)
    
    
def get_nameless_entity(dokuman,indices_only = False):

    '''Girdi olarak yalnizca spaCy dokumani kabul eder'''
    
    including_adjective = any([True if token.text.lower() in adjectives else False for token in dokuman])
    including_noun = any([multicheck(token.text.lower(),nouns) for token in dokuman])
    
    if including_noun and including_adjective:
        adjective_indices = [indis for indis,token in enumerate(dokuman) if token.text.lower() in adjectives]
        noun_indices = [indis for indis,token in enumerate(dokuman) if  multicheck(token.text.lower(),nouns)]
    
        index_combinations = list(itertools.product(adjective_indices, noun_indices))
        index_combinations = pd.DataFrame(index_combinations,columns = ['adj','noun'])
        
        rows_ = []
        for token in dokuman:
            row_ = pd.DataFrame([{'adj':token.i, 'dep': token.dep_, 'noun':token.head.i}])
            rows_.append(row_)
        required_dependencies = pd.concat(rows_).query(f'dep == "amod" or dep == "det"')[['adj','noun']]
        
        index_combinations = index_combinations.merge(required_dependencies,on = ['adj','noun'],how = 'inner')
        
        index_combinations = index_combinations.query(f'noun > adj')
    
        index_combinations['diff'] = index_combinations['noun'] - index_combinations['adj']
    
        index_combinations = index_combinations.sort_values('diff',ascending = True)
    
        selected_index = index_combinations.head(1)

        if len(selected_index) > 0:
            adj_index = selected_index['adj'].item()
        
            noun_index = selected_index['noun'].item()
            
            if coreference_resolver(candidate=dokuman[adj_index:noun_index+1].text,sentence=dokuman.text):
                return None
            else:
                if indices_only:
                    return (adj_index,noun_index)
                else:
                    return dokuman[adj_index:noun_index+1]
        else:
            return None
    else:
        return None