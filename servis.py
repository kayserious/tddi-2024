"""
COK ONEMLI NOT:
SERVIS KODU ICIN TEK DOSYA YUKLEMEYE IZIN VERILDIGI ICIN servis.py DOSYASINDA
IMPORT EDILEN tddi_model VE ner_postprocess_pipeline MODULLERI YUKLENMEMISTIR

"""



import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel, Field
from tddi_model import TDDIModel
from ner_postprocess_pipeline import post_process_ner

SPACY_MODEL_PATH = 'tr_core_news_trf_post_process'
ASPECT_MODEL_PATH = 'tddi-aspect-model'
POLARITY_MODEL_PATH = 'tddi-polarity-model'


lv_model = TDDIModel(load = True,spacy_model = SPACY_MODEL_PATH,aspect_model = ASPECT_MODEL_PATH,polarity_model = POLARITY_MODEL_PATH)

app = FastAPI()

class Item(BaseModel):
    text: str = Field(..., example="""Fiber 100mb SuperOnline kullanıcısıyım yaklaşık 2 haftadır @Twitch @Kick_Turkey gibi canlı yayın platformlarında 360p yayın izlerken donmalar yaşıyoruz.  Başka hiç bir operatörler bu sorunu yaşamazken ben parasını verip alamadığım hizmeti neden ödeyeyim ? @Turkcell """)

@app.post("/predict/", response_model=dict)
async def predict(item: Item):

    result = lv_model.predict_sentence(item.text)
    
    return result


if __name__=="__main__":
    uvicorn.run(app,host="0.0.0.0",port=8000)