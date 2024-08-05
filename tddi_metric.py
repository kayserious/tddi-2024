import pandas as pd
import numpy as np

def tddi_metric(y_true,y_pred,sentiment_coef = 0.65, word_coef = 0.35):
    
    def compare_sentiments(y_true,y_pred):
        
        true_results = pd.DataFrame(y_true['results']).add_prefix('obs_')
        pred_results = pd.DataFrame(y_pred['results']).add_prefix('pred_')
        
        merged = true_results.merge(pred_results,left_on = 'obs_entity',right_on = 'pred_entity',how = 'left')
        
        merged = merged[['obs_sentiment','pred_sentiment']].dropna()
        
        merged['match'] = (merged['obs_sentiment'] == merged['pred_sentiment']).astype(int)
        
        return merged['match'].sum()
        
    
    true_entities = y_true['entity_list']
    
    pred_entities = y_pred['entity_list']
    
    word_denominator = np.maximum(len(true_entities),len(pred_entities))
    
    found_words = len(set(pred_entities).intersection(set(true_entities)))
    
    found_sentiments = compare_sentiments(y_true,y_pred)
    
    word_score = word_coef*(found_words/word_denominator)
    
    sentiment_score = sentiment_coef*(found_sentiments/word_denominator)
    
    return sentiment_score + word_score