**Model**

Model eğitimi ve eğitilmiş bir modeli canlıda kullanmak için;

    
    
    from tddi_model import TDDIModel
    
    from ner_postprocess_pipeline import post_process_ner
    
    

**Model Eğitimi**

Model arayüzü geliştirilirken sklearn API ile uyumlu çalışacak şekilde düzenlenmiştir (.fit() ve .predict() metotları)

X ve y olmak üzere iki `pandas.Series` objesini ve bir doğrulama seti (opsiyonel) girdi olarak alır



    from tddi_model import TDDIModel
    
    from ner_postprocess_pipeline import post_process_ner
    
    df = pd.DataFrame({‘comments’:comment_list,’y’:output_list})
    
    X = df[‘Comments’]
    
    y = df[‘y’]
    
    model = TDDIModel(spacy_model = ‘tr_core_news_trf_post_process’)
    
    model.fit(X = X,y = y,eval_X = None,eval_y = None)
    
    model.model.save_pretrained(
    
    "tddi-aspect-model",
    
    "tddi-polarity-model",
    
    )
    
    

Bu aşamadan sonra model istenilirse canlıda kullanılabilir

**Model Kullanımı**

Kaydedilen aspect, polarity ve spaCy modelleri ile  `load=True` argümanıyla model tekrardan ayağa kaldırılabilir;


    from tddi_model import TDDIModel
    
    from ner_postprocess_pipeline import post_process_ner
    
    SPACY_MODEL_PATH = 'tr_core_news_trf_post_process'
    
    ASPECT_MODEL_PATH = 'tddi-aspect-model'
    
    POLARITY_MODEL_PATH = 'tddi-polarity-model'
    
    model = TDDIModel(load = True,spacy_model = SPACY_MODEL_PATH,aspect_model = ASPECT_MODEL_PATH,polarity_model = POLARITY_MODEL_PATH)
    
    model.predict_sentence(ornek_cumle)
    
**Geliştirme Süreci**

| Sentiment + NER Modeli                                          | CV | Eğitim Epoch | Eğitim TDDI Score | Doğrulama TDDI Score | Not                                                                                                                                                                                                               |
|-----------------------------------------------------------------|---:|--------------|-------------------|----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 'dbmdz/bert-base-turkish-cased' + spacy trf extended            | 5  | 5            | 47.44%            | 48.10%               |                                                                                                                                                                                                                   |
| 'savasy/bert-base-turkish-sentiment-cased' + spacy trf extended | 5  | 5            | 47.27%            | 48.39%               |                                                                                                                                                                                                                   |
| 'savasy/bert-base-turkish-sentiment-cased' + spacy trf extended | 5  | 5            | 49.46%            | 50.05%               | removing punc only + capitalization to   ner plus pipeline                                                                                                                                                        |
| 'savasy/bert-base-turkish-sentiment-cased' + spacy trf extended | 5  | 7            | 51.54%            | 50.17%               | removing punc only + capitalization to   ner plus pipeline (kaggle kernel)                                                                                                                                        |
| 'savasy/bert-base-turkish-sentiment-cased' + spacy trf extended | 5  | 7            | 68.47%            | 65.88%               | nameless entities removed + removing punc   only + capitalization to ner plus pipeline (kaggle kernel)                                                                                                            |
| 'savasy/bert-base-turkish-sentiment-cased' + spacy trf extended | 5  | 5            | 67.16%            | 67.23%               | nameless entities removed + removing punc   only + capitalization to ner plus pipeline                                                                                                                            |
| 'loodos/bert-base-turkish-uncased' + spacy trf extended         | 5  | 5            | 71.26%            | 71.67%               | percent label removed +nameless entities   removed + removing punc only + capitalization to ner plus pipeline                                                                                                     |
| 'loodos/bert-base-turkish-uncased' + spacy trf extended         | 5  | 7            | 77.07%            | 75.50%               | percent label removed +nameless entities   removed + removing punc only + capitalization to ner plus pipeline (kaggle   kernel)                                                                                   |
| 'loodos/bert-base-turkish-uncased' + spacy trf extended         | 5  | 5            | 78.12%            | 76.61%               | percent label removed +nameless entities   removed + nameless entities detected by in sentence patterns and coreference   resolution + removing punc only + capitalization to ner plus pipeline (kaggle   kernel) |
| 'loodos/bert-base-turkish-uncased' + spacy trf extended         | 5  | 7            | 78.63%            | 76.98%               | percent label removed +nameless entities   removed + nameless entities detected by in sentence patterns and coreference   resolution + removing punc only + capitalization to ner plus pipeline (kaggle   kernel) |
| 'loodos/bert-base-turkish-uncased' + spacy trf extended         | 5  | 7            | 82.28%            | 82.10%               | percent label removed +nameless entities   removed + nameless entities detected by in sentence patterns and coreference   resolution + removing punc only + capitalization to ner plus pipeline (kaggle   kernel) |