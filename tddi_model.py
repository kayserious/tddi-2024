from setfit import AbsaModel,AbsaTrainer,TrainingArguments
from datasets import Dataset
import setfit as sf

import pandas as pd

def get_absa_dataset(df):
    dff = df.copy()
    dff['absa'] = dff['y'].apply(lambda x : x['results'])
    dff = dff.explode('absa')
    dff['span'] = dff['absa'].apply(lambda x : x['entity'])
    dff['label'] = dff['absa'].apply(lambda x : x['sentiment'])
    dff = dff.rename(columns = {'comments':'text'})
    dff = dff[['text','span','label']].assign(ordinal = 0)
    return dff

class TDDIModel:
    def __init__(self,spacy_model,sentiment_model=None,num_epoch = 5,batch_size = 40,output_dir = "D:\\HF\\absa",load = False,aspect_model = None,polarity_model = None):
        if load:
            self.model = AbsaModel.from_pretrained(aspect_model,polarity_model,spacy_model=spacy_model)
        else:
            self.model = AbsaModel.from_pretrained(sentiment_model,spacy_model=spacy_model)
        self.output_dir = output_dir
        self.num_epoch = num_epoch
        self.batch_size = batch_size

    def fit(self,X,y,eval_X = None,eval_y = None):
    
        if eval_X is None or eval_y is None:
            will_be_evaluated = False
        else:
            will_be_evaluated = True
            
        self.args = sf.TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            num_epochs = self.num_epoch,
            batch_size = self.batch_size,
            sampling_strategy = "undersampling"
        )

        self.args.eval_strategy = self.args.evaluation_strategy

        self.trn_absa = get_absa_dataset(pd.concat([X,y],axis = 1))
        self.trn_absa = Dataset.from_pandas(self.trn_absa)
        
        if will_be_evaluated:
        
            self.val_absa = get_absa_dataset(pd.concat([eval_X,eval_y],axis = 1))
            self.val_absa = Dataset.from_pandas(self.val_absa)
            
        if will_be_evaluated:

            self.trainer = AbsaTrainer(
                self.model,
                args=self.args,
                train_dataset=self.trn_absa,
                eval_dataset=self.val_absa
            )
            
        else:
        
            self.trainer = AbsaTrainer(
                self.model,
                args=self.args,
                train_dataset=self.trn_absa
            )
        #callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        self.trainer.train()

    def predict(self,XOXO):

        self.Xlist = pd.DataFrame(XOXO).rename(columns = {'comments':'text'})['text'].tolist()

        self.raw_preds = self.model.predict(self.Xlist)

        def singular_preds(absa_raw_pred):
            opts = list(set([i['span'] for i in absa_raw_pred]))
            rnw = []
            for option in opts:
                rnw.append([i for i in absa_raw_pred if i['span'] == option][0])
            return rnw

        self.raw_preds = [singular_preds(jj) for jj in self.raw_preds]

        self.preds = [{'entity_list':[j['span'] for j in i],'results':[{'entity':j['span'],'sentiment':j['polarity']} for j in i]} for i in self.raw_preds]

        return self.preds
        
        
    def predict_sentence(self,sentence):
    
        to_model_sent = pd.DataFrame([{'comments':sentence}])
        
        model_returns = self.predict(to_model_sent)
        
        return model_returns[0]