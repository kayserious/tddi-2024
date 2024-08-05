import pandas as pd
import ast

def load_competition_data(dpath):

    data = pd.read_excel(dpath)
    
    corrected_data = []
    for _,i in data.iterrows():
        dicto = ast.literal_eval(i['duygu']).keys()
        if len(dicto) == 0:
            continue
        if list(dicto)[0].strip() == '':
            continue
        
        corrected_data.append(pd.DataFrame([{'comments':i['comments'],'duygu':i['duygu']}]))
        
    data = pd.concat(corrected_data).reset_index(drop = True)
    
    def extract_y(xx):
        entities = list(ast.literal_eval(xx).keys())
        entity_results = [{'entity':i,'sentiment':j} for i,j in zip(ast.literal_eval(xx).keys(),ast.literal_eval(xx).values())]
        return {'entity_list':entities,'results':entity_results}

    data['y'] = data['duygu'].apply(extract_y)
    
    return data