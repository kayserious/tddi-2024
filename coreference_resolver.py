tfidf_for_coref_path = 'custom coreference resolver/tfidf_for_coref.pkl'
tfidf_for_sentence_path = 'custom coreference resolver/tfidf_for_sentence.pkl'
coref_xgb_model_path = 'custom coreference resolver/coreference_resolver_xgb.mdl'

from xgboost import XGBClassifier
import joblib
import pandas as pd

tfidf_for_coref = joblib.load(tfidf_for_coref_path)
tfidf_for_sentence = joblib.load(tfidf_for_sentence_path)
coref_model = XGBClassifier()
coref_model.load_model(coref_xgb_model_path)

def coreference_resolver(candidate,sentence):
    coref_df = pd.DataFrame([{'coref':candidate}])
    sentence_df = pd.DataFrame([{'sentence':sentence}])
    
    train_sentences = tfidf_for_sentence.transform(sentence_df['sentence']).toarray()
    train_sentences = pd.DataFrame(train_sentences).add_prefix('sentence_')

    train_corefs = tfidf_for_coref.transform(coref_df['coref']).toarray()
    train_corefs = pd.DataFrame(train_corefs).add_prefix('coref_')
    
    train_vectorized = pd.concat([train_sentences,train_corefs],axis = 1)
    
    return bool(coref_model.predict(train_vectorized).item())