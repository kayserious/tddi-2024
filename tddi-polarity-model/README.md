---
base_model: loodos/bert-base-turkish-uncased
library_name: setfit
metrics:
- accuracy
pipeline_tag: text-classification
tags:
- setfit
- absa
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
widget:
- text: Vodafone faturalı hattımı faturasıza:Vodafone faturalı hattımı faturasıza
    geçirmek istiyorum. Müşteri hizmetlerine saatlerce bağlanamıyorum. Bağlansam da
    işim çözülmüyor. Birleşmiş milletler daha rahat ulaşırım. Alakasız temsilci ancak
    konuşuyor, sonra da sizi arayacağız diyorlar. Bu kadar vurdumduymaz bir şirket
    olamaz.
- text: Decathlon aldığım kamp malzemeleri:Decathlon aldığım kamp malzemeleri ile
    harika bir doğa deneyimi yaşadım. Özellikle çadır çok kullanışlı ve kolay kurulumu
    var. Uyku tulumu da çok rahat ve sıcak tuttu. Fiyatları da oldukça uygun. Outdoor
    aktiviteler için kaliteli ve ekonomik ürünler arıyorsanız kesinlikle tavsiye ederim.
- text: Cildim için Nivea yeni nemlendiricisini aldım:Cildim için Nivea yeni nemlendiricisini
    aldım ama hiç iyi gelmedi. Cildim kurudu ve kaşıntı yaptı, sanırım içeriği bana
    uygun değil.
- text: IKEA aldığım mobilyaların montajı:IKEA aldığım mobilyaların montajı çok zor
    ve talimatlar yetersiz. Ayrıca, mobilyaların kalitesi de beklediğimden düşük çıktı.
    Bazı parçalar eksik geldi ve müşteri hizmetleri de bu konuda yardımcı olmadı.
    IKEA'dan tekrar alışveriş yapmayı düşünmüyorum.
- text: Eve TurkNet bağlattığımdan beri problem:Eve TurkNet bağlattığımdan beri problem
    yaşıyorum. Sürekli internet kesintilerinin bir yana bunun mağduriyeti ve çözümü
    için hiçbir şey yapmıyorlar. Yaklaşık 4 gündür internetim yok ve defalarca müşteri
    hizmetlerini arayıp arıza kaydı açtırmama rağmen ekip asla randevu kaydı için
    bile geri aramıyor.
inference: false
---

# SetFit Polarity Model with loodos/bert-base-turkish-uncased

This is a [SetFit](https://github.com/huggingface/setfit) model that can be used for Aspect Based Sentiment Analysis (ABSA). This SetFit model uses [loodos/bert-base-turkish-uncased](https://huggingface.co/loodos/bert-base-turkish-uncased) as the Sentence Transformer embedding model. A [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance is used for classification. In particular, this model is in charge of classifying aspect polarities.

The model has been trained using an efficient few-shot learning technique that involves:

1. Fine-tuning a [Sentence Transformer](https://www.sbert.net) with contrastive learning.
2. Training a classification head with features from the fine-tuned Sentence Transformer.

This model was trained within the context of a larger system for ABSA, which looks like so:

1. Use a spaCy model to select possible aspect span candidates.
2. Use a SetFit model to filter these possible aspect span candidates.
3. **Use this SetFit model to classify the filtered aspect span candidates.**

## Model Details

### Model Description
- **Model Type:** SetFit
- **Sentence Transformer body:** [loodos/bert-base-turkish-uncased](https://huggingface.co/loodos/bert-base-turkish-uncased)
- **Classification head:** a [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance
- **spaCy Model:** tr_core_news_trf_post_process
- **SetFitABSA Aspect Model:** [setfit-absa-aspect](https://huggingface.co/setfit-absa-aspect)
- **SetFitABSA Polarity Model:** [tddi-polarity-model](https://huggingface.co/tddi-polarity-model)
- **Maximum Sequence Length:** 512 tokens
- **Number of Classes:** 3 classes
<!-- - **Training Dataset:** [Unknown](https://huggingface.co/datasets/unknown) -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Repository:** [SetFit on GitHub](https://github.com/huggingface/setfit)
- **Paper:** [Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055)
- **Blogpost:** [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)

### Model Labels
| Label   | Examples                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|:--------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| nötr    | <ul><li>'kullanmadığım sac serum markası kalmadi diyebilirim ...:kullanmadığım sac serum markası kalmadi diyebilirim ...Ancak bioxin gerçekten çok başarılı saçlarım uzun ve gür agirlastirmadan yumuşacık yapıyor...sadece kokusu çok iyi değil umarım onun içinde bir formül bulunur .... stok yapılabilecek şekilde başarılı fiyatı gramajina göre çok iyi bence...'</li><li>'ile alakalıdır diye Lenovo sitesinden tüm güncellemeleri:ürün anlatıldığı gibi şık ve kibar bir tasarıma sahip. eksi yanı tek hoparlör var veya biri çalışmıyor. sol taraftan ses geliyor sağ taraf boş kalıyor buda bişe izlerken rahatsız edici bir durum. driver ile alakalıdır diye Lenovo sitesinden tüm güncellemeleri yaptım ama aynı'</li><li>'. Konuyu Tüket Hakem Heyeti ve CİMER de taşıyacağım.:TAB Gıda bağlı\xa0Popeyes\xa0markalı İzmir Optimum AVM bulunan restoranda sipariş verdim. Siparişimi kasa arkasında bulunan ve herkese yayınlanmakta olan büyük ekranlı dijital menüden fiyatı 190 TL olarak gösterilen bir sandviç menü olarak seçtim. Personel siparişimi alıp kasasında\xa0işlediğinde ise 200 TL olarak bir ödeme istedi. Konuyu Tüket Hakem Heyeti ve CİMER de taşıyacağım.'</li></ul> |
| olumlu  | <ul><li>'diyebilirim ...Ancak bioxin gerçekten çok başarılı:kullanmadığım sac serum markası kalmadi diyebilirim ...Ancak bioxin gerçekten çok başarılı saçlarım uzun ve gür agirlastirmadan yumuşacık yapıyor...sadece kokusu çok iyi değil umarım onun içinde bir formül bulunur .... stok yapılabilecek şekilde başarılı fiyatı gramajina göre çok iyi bence...'</li><li>'Loreal bioblas yağlarını kullanıyordum benim:Loreal bioblas yağlarını kullanıyordum benim saçlarım kuru ve sarıya boyatıyorum sürekli saçıma normalde bayağı sürmem gerekiyor kuru olduğu için ama çok az sürmeme rağmen nemlendirdi parlattı yumuşacık yaptı yağlı saçlar sanırım 1 yılda anca bitirir bu yağı ben çok beğendim kuru saçlar öncelikli almalı bence'</li><li>'kaldım. Birde trendyol un hızlı kargosu:Bilgisayar tamiri yapan bir arkadaşıma danışarak aldım. işlemcisi olmasının performansa dayalı olduğunu söyleyince indirimden hemen aldım. İnce ,hafif ,muazzam hızlı ,tam klavye ofiste kullanmak için biçilmiş kaftan. Çok memnun kaldım. Birde trendyol un hızlı kargosu teşekkürler.'</li></ul>                                                                                                    |
| olumsuz | <ul><li>'da beğendi. diğer p.. markası hindistan cevizli yağıyla:1 yıldır kullanıyorum. saçı ağırlaştırmadan nemlendiriyor ve saçım öncesine nazaran daha canlı (boyalı saçım) halam için aldım bunu da o da beğendi. diğer p.. markası hindistan cevizli yağıyla karşılaştırınca ne kadar hafif bir yapısı olduğunu fark ettim. önerilir ❤️'</li><li>'Zara yaklaşık 2 hafta:Zara yaklaşık 2 hafta önce bir parfüm satın aldım\xa0online\xa0dan alışveriş yaptım tahmini\xa0teslimat tarihi yarındı ancak bugün\xa0siparişim iptal edildi. Madem stok yok neden online koyuyorsunuz 14 gün boyunca paramı elinizde tutup edip iptal ediyorsunuz. İptal işlemini\xa0kabul\xa0etmiyorum\xa0ürünümü göndermenizi istiyorum'</li><li>'TAB Gıda bağlı\xa0Popeyes:TAB Gıda bağlı\xa0Popeyes\xa0markalı İzmir Optimum AVM bulunan restoranda sipariş verdim. Siparişimi kasa arkasında bulunan ve herkese yayınlanmakta olan büyük ekranlı dijital menüden fiyatı 190 TL olarak gösterilen bir sandviç menü olarak seçtim. Personel siparişimi alıp kasasında\xa0işlediğinde ise 200 TL olarak bir ödeme istedi. Konuyu Tüket Hakem Heyeti ve CİMER de taşıyacağım.'</li></ul>                                  |

## Uses

### Direct Use for Inference

First install the SetFit library:

```bash
pip install setfit
```

Then you can load this model and run inference.

```python
from setfit import AbsaModel

# Download from the 🤗 Hub
model = AbsaModel.from_pretrained(
    "setfit-absa-aspect",
    "tddi-polarity-model",
)
# Run inference
preds = model("The food was great, but the venue is just way too busy.")
```

<!--
### Downstream Use

*List how someone could finetune this model on their own dataset.*
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Set Metrics
| Training set | Min | Median  | Max |
|:-------------|:----|:--------|:----|
| Word count   | 10  | 41.7765 | 99  |

| Label   | Training Sample Count |
|:--------|:----------------------|
| nötr    | 37                    |
| olumlu  | 107                   |
| olumsuz | 393                   |

### Training Hyperparameters
- batch_size: (40, 40)
- num_epochs: (7, 7)
- max_steps: -1
- sampling_strategy: undersampling
- body_learning_rate: (2e-05, 1e-05)
- head_learning_rate: 0.01
- loss: CosineSimilarityLoss
- distance_metric: cosine_distance
- margin: 0.25
- end_to_end: False
- use_amp: False
- warmup_proportion: 0.1
- seed: 42
- eval_max_steps: -1
- load_best_model_at_end: False

### Training Results
| Epoch  | Step  | Training Loss | Validation Loss |
|:------:|:-----:|:-------------:|:---------------:|
| 0.0003 | 1     | 0.3587        | -               |
| 0.0165 | 50    | 0.3311        | -               |
| 0.0330 | 100   | 0.2817        | -               |
| 0.0495 | 150   | 0.1655        | -               |
| 0.0661 | 200   | 0.1976        | -               |
| 0.0826 | 250   | 0.1536        | -               |
| 0.0991 | 300   | 0.1524        | -               |
| 0.1156 | 350   | 0.0993        | -               |
| 0.1321 | 400   | 0.0755        | -               |
| 0.1486 | 450   | 0.0223        | -               |
| 0.1651 | 500   | 0.0199        | -               |
| 0.1816 | 550   | 0.0141        | -               |
| 0.1982 | 600   | 0.0016        | -               |
| 0.2147 | 650   | 0.0025        | -               |
| 0.2312 | 700   | 0.0006        | -               |
| 0.2477 | 750   | 0.0006        | -               |
| 0.2642 | 800   | 0.0004        | -               |
| 0.2807 | 850   | 0.0008        | -               |
| 0.2972 | 900   | 0.0001        | -               |
| 0.3137 | 950   | 0.0005        | -               |
| 0.3303 | 1000  | 0.001         | -               |
| 0.3468 | 1050  | 0.0001        | -               |
| 0.3633 | 1100  | 0.0021        | -               |
| 0.3798 | 1150  | 0.0003        | -               |
| 0.3963 | 1200  | 0.0001        | -               |
| 0.4128 | 1250  | 0.0001        | -               |
| 0.4293 | 1300  | 0.0001        | -               |
| 0.4458 | 1350  | 0.0001        | -               |
| 0.4624 | 1400  | 0.0           | -               |
| 0.4789 | 1450  | 0.0001        | -               |
| 0.4954 | 1500  | 0.0           | -               |
| 0.5119 | 1550  | 0.0001        | -               |
| 0.5284 | 1600  | 0.0001        | -               |
| 0.5449 | 1650  | 0.0001        | -               |
| 0.5614 | 1700  | 0.0           | -               |
| 0.5779 | 1750  | 0.0013        | -               |
| 0.5945 | 1800  | 0.0001        | -               |
| 0.6110 | 1850  | 0.0           | -               |
| 0.6275 | 1900  | 0.0           | -               |
| 0.6440 | 1950  | 0.0001        | -               |
| 0.6605 | 2000  | 0.0           | -               |
| 0.6770 | 2050  | 0.0001        | -               |
| 0.6935 | 2100  | 0.0           | -               |
| 0.7100 | 2150  | 0.0           | -               |
| 0.7266 | 2200  | 0.0           | -               |
| 0.7431 | 2250  | 0.0           | -               |
| 0.7596 | 2300  | 0.0           | -               |
| 0.7761 | 2350  | 0.0           | -               |
| 0.7926 | 2400  | 0.0           | -               |
| 0.8091 | 2450  | 0.0           | -               |
| 0.8256 | 2500  | 0.0           | -               |
| 0.8421 | 2550  | 0.0           | -               |
| 0.8587 | 2600  | 0.0           | -               |
| 0.8752 | 2650  | 0.0           | -               |
| 0.8917 | 2700  | 0.0           | -               |
| 0.9082 | 2750  | 0.0           | -               |
| 0.9247 | 2800  | 0.0           | -               |
| 0.9412 | 2850  | 0.0           | -               |
| 0.9577 | 2900  | 0.0           | -               |
| 0.9742 | 2950  | 0.0           | -               |
| 0.9908 | 3000  | 0.0           | -               |
| 1.0073 | 3050  | 0.0           | -               |
| 1.0238 | 3100  | 0.0           | -               |
| 1.0403 | 3150  | 0.0           | -               |
| 1.0568 | 3200  | 0.0           | -               |
| 1.0733 | 3250  | 0.0           | -               |
| 1.0898 | 3300  | 0.0           | -               |
| 1.1063 | 3350  | 0.0           | -               |
| 1.1229 | 3400  | 0.0           | -               |
| 1.1394 | 3450  | 0.0           | -               |
| 1.1559 | 3500  | 0.0           | -               |
| 1.1724 | 3550  | 0.0           | -               |
| 1.1889 | 3600  | 0.0           | -               |
| 1.2054 | 3650  | 0.0           | -               |
| 1.2219 | 3700  | 0.0           | -               |
| 1.2384 | 3750  | 0.0           | -               |
| 1.2550 | 3800  | 0.0           | -               |
| 1.2715 | 3850  | 0.0           | -               |
| 1.2880 | 3900  | 0.0           | -               |
| 1.3045 | 3950  | 0.0           | -               |
| 1.3210 | 4000  | 0.0           | -               |
| 1.3375 | 4050  | 0.0           | -               |
| 1.3540 | 4100  | 0.0           | -               |
| 1.3705 | 4150  | 0.0           | -               |
| 1.3871 | 4200  | 0.0           | -               |
| 1.4036 | 4250  | 0.0           | -               |
| 1.4201 | 4300  | 0.0           | -               |
| 1.4366 | 4350  | 0.0           | -               |
| 1.4531 | 4400  | 0.0           | -               |
| 1.4696 | 4450  | 0.0           | -               |
| 1.4861 | 4500  | 0.0           | -               |
| 1.5026 | 4550  | 0.0           | -               |
| 1.5192 | 4600  | 0.0           | -               |
| 1.5357 | 4650  | 0.0           | -               |
| 1.5522 | 4700  | 0.0           | -               |
| 1.5687 | 4750  | 0.0           | -               |
| 1.5852 | 4800  | 0.0           | -               |
| 1.6017 | 4850  | 0.0           | -               |
| 1.6182 | 4900  | 0.0           | -               |
| 1.6347 | 4950  | 0.0           | -               |
| 1.6513 | 5000  | 0.0           | -               |
| 1.6678 | 5050  | 0.0           | -               |
| 1.6843 | 5100  | 0.0           | -               |
| 1.7008 | 5150  | 0.0           | -               |
| 1.7173 | 5200  | 0.0           | -               |
| 1.7338 | 5250  | 0.0           | -               |
| 1.7503 | 5300  | 0.0           | -               |
| 1.7668 | 5350  | 0.0           | -               |
| 1.7834 | 5400  | 0.0           | -               |
| 1.7999 | 5450  | 0.0           | -               |
| 1.8164 | 5500  | 0.0           | -               |
| 1.8329 | 5550  | 0.0           | -               |
| 1.8494 | 5600  | 0.0           | -               |
| 1.8659 | 5650  | 0.0           | -               |
| 1.8824 | 5700  | 0.0           | -               |
| 1.8989 | 5750  | 0.0           | -               |
| 1.9155 | 5800  | 0.0           | -               |
| 1.9320 | 5850  | 0.0           | -               |
| 1.9485 | 5900  | 0.0           | -               |
| 1.9650 | 5950  | 0.0           | -               |
| 1.9815 | 6000  | 0.0           | -               |
| 1.9980 | 6050  | 0.0           | -               |
| 2.0145 | 6100  | 0.0           | -               |
| 2.0310 | 6150  | 0.0           | -               |
| 2.0476 | 6200  | 0.0           | -               |
| 2.0641 | 6250  | 0.0           | -               |
| 2.0806 | 6300  | 0.0           | -               |
| 2.0971 | 6350  | 0.0           | -               |
| 2.1136 | 6400  | 0.0           | -               |
| 2.1301 | 6450  | 0.0           | -               |
| 2.1466 | 6500  | 0.0           | -               |
| 2.1631 | 6550  | 0.0           | -               |
| 2.1797 | 6600  | 0.0           | -               |
| 2.1962 | 6650  | 0.0           | -               |
| 2.2127 | 6700  | 0.0           | -               |
| 2.2292 | 6750  | 0.0           | -               |
| 2.2457 | 6800  | 0.0           | -               |
| 2.2622 | 6850  | 0.0           | -               |
| 2.2787 | 6900  | 0.0           | -               |
| 2.2952 | 6950  | 0.0           | -               |
| 2.3118 | 7000  | 0.0           | -               |
| 2.3283 | 7050  | 0.0           | -               |
| 2.3448 | 7100  | 0.0           | -               |
| 2.3613 | 7150  | 0.0           | -               |
| 2.3778 | 7200  | 0.0           | -               |
| 2.3943 | 7250  | 0.0           | -               |
| 2.4108 | 7300  | 0.0           | -               |
| 2.4273 | 7350  | 0.0           | -               |
| 2.4439 | 7400  | 0.0           | -               |
| 2.4604 | 7450  | 0.0           | -               |
| 2.4769 | 7500  | 0.0           | -               |
| 2.4934 | 7550  | 0.0           | -               |
| 2.5099 | 7600  | 0.0           | -               |
| 2.5264 | 7650  | 0.0           | -               |
| 2.5429 | 7700  | 0.0           | -               |
| 2.5594 | 7750  | 0.0           | -               |
| 2.5760 | 7800  | 0.0           | -               |
| 2.5925 | 7850  | 0.0           | -               |
| 2.6090 | 7900  | 0.0           | -               |
| 2.6255 | 7950  | 0.0           | -               |
| 2.6420 | 8000  | 0.0           | -               |
| 2.6585 | 8050  | 0.0           | -               |
| 2.6750 | 8100  | 0.0           | -               |
| 2.6915 | 8150  | 0.0           | -               |
| 2.7081 | 8200  | 0.0           | -               |
| 2.7246 | 8250  | 0.0           | -               |
| 2.7411 | 8300  | 0.0           | -               |
| 2.7576 | 8350  | 0.0           | -               |
| 2.7741 | 8400  | 0.0           | -               |
| 2.7906 | 8450  | 0.0           | -               |
| 2.8071 | 8500  | 0.0           | -               |
| 2.8236 | 8550  | 0.0           | -               |
| 2.8402 | 8600  | 0.0           | -               |
| 2.8567 | 8650  | 0.0           | -               |
| 2.8732 | 8700  | 0.0           | -               |
| 2.8897 | 8750  | 0.0           | -               |
| 2.9062 | 8800  | 0.0           | -               |
| 2.9227 | 8850  | 0.0           | -               |
| 2.9392 | 8900  | 0.0           | -               |
| 2.9557 | 8950  | 0.0           | -               |
| 2.9723 | 9000  | 0.0           | -               |
| 2.9888 | 9050  | 0.0           | -               |
| 3.0053 | 9100  | 0.0           | -               |
| 3.0218 | 9150  | 0.0           | -               |
| 3.0383 | 9200  | 0.0           | -               |
| 3.0548 | 9250  | 0.0           | -               |
| 3.0713 | 9300  | 0.0           | -               |
| 3.0878 | 9350  | 0.0           | -               |
| 3.1044 | 9400  | 0.0           | -               |
| 3.1209 | 9450  | 0.0           | -               |
| 3.1374 | 9500  | 0.0           | -               |
| 3.1539 | 9550  | 0.0           | -               |
| 3.1704 | 9600  | 0.0           | -               |
| 3.1869 | 9650  | 0.0           | -               |
| 3.2034 | 9700  | 0.0           | -               |
| 3.2199 | 9750  | 0.0           | -               |
| 3.2365 | 9800  | 0.0           | -               |
| 3.2530 | 9850  | 0.0           | -               |
| 3.2695 | 9900  | 0.0           | -               |
| 3.2860 | 9950  | 0.0           | -               |
| 3.3025 | 10000 | 0.0           | -               |
| 3.3190 | 10050 | 0.0           | -               |
| 3.3355 | 10100 | 0.0           | -               |
| 3.3520 | 10150 | 0.0           | -               |
| 3.3686 | 10200 | 0.0           | -               |
| 3.3851 | 10250 | 0.0           | -               |
| 3.4016 | 10300 | 0.0           | -               |
| 3.4181 | 10350 | 0.0           | -               |
| 3.4346 | 10400 | 0.0           | -               |
| 3.4511 | 10450 | 0.0           | -               |
| 3.4676 | 10500 | 0.0           | -               |
| 3.4841 | 10550 | 0.0           | -               |
| 3.5007 | 10600 | 0.0           | -               |
| 3.5172 | 10650 | 0.0           | -               |
| 3.5337 | 10700 | 0.0           | -               |
| 3.5502 | 10750 | 0.0           | -               |
| 3.5667 | 10800 | 0.0           | -               |
| 3.5832 | 10850 | 0.0           | -               |
| 3.5997 | 10900 | 0.0           | -               |
| 3.6162 | 10950 | 0.0           | -               |
| 3.6328 | 11000 | 0.0           | -               |
| 3.6493 | 11050 | 0.0           | -               |
| 3.6658 | 11100 | 0.0           | -               |
| 3.6823 | 11150 | 0.0           | -               |
| 3.6988 | 11200 | 0.0           | -               |
| 3.7153 | 11250 | 0.0           | -               |
| 3.7318 | 11300 | 0.0           | -               |
| 3.7483 | 11350 | 0.0           | -               |
| 3.7649 | 11400 | 0.0           | -               |
| 3.7814 | 11450 | 0.0           | -               |
| 3.7979 | 11500 | 0.0           | -               |
| 3.8144 | 11550 | 0.0           | -               |
| 3.8309 | 11600 | 0.0           | -               |
| 3.8474 | 11650 | 0.0           | -               |
| 3.8639 | 11700 | 0.0           | -               |
| 3.8804 | 11750 | 0.0           | -               |
| 3.8970 | 11800 | 0.0           | -               |
| 3.9135 | 11850 | 0.0           | -               |
| 3.9300 | 11900 | 0.0           | -               |
| 3.9465 | 11950 | 0.0           | -               |
| 3.9630 | 12000 | 0.0           | -               |
| 3.9795 | 12050 | 0.0           | -               |
| 3.9960 | 12100 | 0.0           | -               |
| 4.0125 | 12150 | 0.0           | -               |
| 4.0291 | 12200 | 0.0           | -               |
| 4.0456 | 12250 | 0.0           | -               |
| 4.0621 | 12300 | 0.0           | -               |
| 4.0786 | 12350 | 0.0275        | -               |
| 4.0951 | 12400 | 0.0005        | -               |
| 4.1116 | 12450 | 0.0001        | -               |
| 4.1281 | 12500 | 0.0002        | -               |
| 4.1446 | 12550 | 0.0           | -               |
| 4.1612 | 12600 | 0.0001        | -               |
| 4.1777 | 12650 | 0.0001        | -               |
| 4.1942 | 12700 | 0.0           | -               |
| 4.2107 | 12750 | 0.0           | -               |
| 4.2272 | 12800 | 0.0001        | -               |
| 4.2437 | 12850 | 0.0           | -               |
| 4.2602 | 12900 | 0.0           | -               |
| 4.2768 | 12950 | 0.0           | -               |
| 4.2933 | 13000 | 0.0           | -               |
| 4.3098 | 13050 | 0.0           | -               |
| 4.3263 | 13100 | 0.0           | -               |
| 4.3428 | 13150 | 0.0           | -               |
| 4.3593 | 13200 | 0.0           | -               |
| 4.3758 | 13250 | 0.0           | -               |
| 4.3923 | 13300 | 0.0           | -               |
| 4.4089 | 13350 | 0.0           | -               |
| 4.4254 | 13400 | 0.0           | -               |
| 4.4419 | 13450 | 0.0           | -               |
| 4.4584 | 13500 | 0.0           | -               |
| 4.4749 | 13550 | 0.0           | -               |
| 4.4914 | 13600 | 0.0           | -               |
| 4.5079 | 13650 | 0.0           | -               |
| 4.5244 | 13700 | 0.0           | -               |
| 4.5410 | 13750 | 0.0           | -               |
| 4.5575 | 13800 | 0.0           | -               |
| 4.5740 | 13850 | 0.0           | -               |
| 4.5905 | 13900 | 0.0           | -               |
| 4.6070 | 13950 | 0.0           | -               |
| 4.6235 | 14000 | 0.0           | -               |
| 4.6400 | 14050 | 0.0           | -               |
| 4.6565 | 14100 | 0.0           | -               |
| 4.6731 | 14150 | 0.0           | -               |
| 4.6896 | 14200 | 0.0           | -               |
| 4.7061 | 14250 | 0.0           | -               |
| 4.7226 | 14300 | 0.0           | -               |
| 4.7391 | 14350 | 0.0           | -               |
| 4.7556 | 14400 | 0.0           | -               |
| 4.7721 | 14450 | 0.0           | -               |
| 4.7886 | 14500 | 0.0           | -               |
| 4.8052 | 14550 | 0.0           | -               |
| 4.8217 | 14600 | 0.0           | -               |
| 4.8382 | 14650 | 0.0           | -               |
| 4.8547 | 14700 | 0.0           | -               |
| 4.8712 | 14750 | 0.0           | -               |
| 4.8877 | 14800 | 0.0           | -               |
| 4.9042 | 14850 | 0.0           | -               |
| 4.9207 | 14900 | 0.0           | -               |
| 4.9373 | 14950 | 0.0           | -               |
| 4.9538 | 15000 | 0.0           | -               |
| 4.9703 | 15050 | 0.0           | -               |
| 4.9868 | 15100 | 0.0           | -               |
| 5.0033 | 15150 | 0.0           | -               |
| 5.0198 | 15200 | 0.0           | -               |
| 5.0363 | 15250 | 0.0           | -               |
| 5.0528 | 15300 | 0.0           | -               |
| 5.0694 | 15350 | 0.0           | -               |
| 5.0859 | 15400 | 0.0           | -               |
| 5.1024 | 15450 | 0.0           | -               |
| 5.1189 | 15500 | 0.0           | -               |
| 5.1354 | 15550 | 0.0           | -               |
| 5.1519 | 15600 | 0.0           | -               |
| 5.1684 | 15650 | 0.0           | -               |
| 5.1849 | 15700 | 0.0           | -               |
| 5.2015 | 15750 | 0.0           | -               |
| 5.2180 | 15800 | 0.0           | -               |
| 5.2345 | 15850 | 0.0           | -               |
| 5.2510 | 15900 | 0.0           | -               |
| 5.2675 | 15950 | 0.0           | -               |
| 5.2840 | 16000 | 0.0           | -               |
| 5.3005 | 16050 | 0.0           | -               |
| 5.3170 | 16100 | 0.0           | -               |
| 5.3336 | 16150 | 0.0           | -               |
| 5.3501 | 16200 | 0.0           | -               |
| 5.3666 | 16250 | 0.0           | -               |
| 5.3831 | 16300 | 0.0           | -               |
| 5.3996 | 16350 | 0.0           | -               |
| 5.4161 | 16400 | 0.0           | -               |
| 5.4326 | 16450 | 0.0           | -               |
| 5.4491 | 16500 | 0.0           | -               |
| 5.4657 | 16550 | 0.0           | -               |
| 5.4822 | 16600 | 0.0           | -               |
| 5.4987 | 16650 | 0.0           | -               |
| 5.5152 | 16700 | 0.0           | -               |
| 5.5317 | 16750 | 0.0           | -               |
| 5.5482 | 16800 | 0.0           | -               |
| 5.5647 | 16850 | 0.0           | -               |
| 5.5812 | 16900 | 0.0           | -               |
| 5.5978 | 16950 | 0.0           | -               |
| 5.6143 | 17000 | 0.0           | -               |
| 5.6308 | 17050 | 0.0           | -               |
| 5.6473 | 17100 | 0.0           | -               |
| 5.6638 | 17150 | 0.0           | -               |
| 5.6803 | 17200 | 0.0           | -               |
| 5.6968 | 17250 | 0.0           | -               |
| 5.7133 | 17300 | 0.0           | -               |
| 5.7299 | 17350 | 0.0           | -               |
| 5.7464 | 17400 | 0.0           | -               |
| 5.7629 | 17450 | 0.0           | -               |
| 5.7794 | 17500 | 0.0           | -               |
| 5.7959 | 17550 | 0.0           | -               |
| 5.8124 | 17600 | 0.0           | -               |
| 5.8289 | 17650 | 0.0           | -               |
| 5.8454 | 17700 | 0.0           | -               |
| 5.8620 | 17750 | 0.0           | -               |
| 5.8785 | 17800 | 0.0           | -               |
| 5.8950 | 17850 | 0.0           | -               |
| 5.9115 | 17900 | 0.0           | -               |
| 5.9280 | 17950 | 0.0           | -               |
| 5.9445 | 18000 | 0.0           | -               |
| 5.9610 | 18050 | 0.0           | -               |
| 5.9775 | 18100 | 0.0           | -               |
| 5.9941 | 18150 | 0.0           | -               |
| 6.0106 | 18200 | 0.0           | -               |
| 6.0271 | 18250 | 0.0           | -               |
| 6.0436 | 18300 | 0.0           | -               |
| 6.0601 | 18350 | 0.0           | -               |
| 6.0766 | 18400 | 0.0           | -               |
| 6.0931 | 18450 | 0.0           | -               |
| 6.1096 | 18500 | 0.0           | -               |
| 6.1262 | 18550 | 0.0           | -               |
| 6.1427 | 18600 | 0.0           | -               |
| 6.1592 | 18650 | 0.0           | -               |
| 6.1757 | 18700 | 0.0           | -               |
| 6.1922 | 18750 | 0.0           | -               |
| 6.2087 | 18800 | 0.0           | -               |
| 6.2252 | 18850 | 0.0           | -               |
| 6.2417 | 18900 | 0.0           | -               |
| 6.2583 | 18950 | 0.0           | -               |
| 6.2748 | 19000 | 0.0           | -               |
| 6.2913 | 19050 | 0.0           | -               |
| 6.3078 | 19100 | 0.0           | -               |
| 6.3243 | 19150 | 0.0           | -               |
| 6.3408 | 19200 | 0.0           | -               |
| 6.3573 | 19250 | 0.0           | -               |
| 6.3738 | 19300 | 0.0           | -               |
| 6.3904 | 19350 | 0.0           | -               |
| 6.4069 | 19400 | 0.0           | -               |
| 6.4234 | 19450 | 0.0           | -               |
| 6.4399 | 19500 | 0.0           | -               |
| 6.4564 | 19550 | 0.0           | -               |
| 6.4729 | 19600 | 0.0           | -               |
| 6.4894 | 19650 | 0.0           | -               |
| 6.5059 | 19700 | 0.0           | -               |
| 6.5225 | 19750 | 0.0           | -               |
| 6.5390 | 19800 | 0.0           | -               |
| 6.5555 | 19850 | 0.0           | -               |
| 6.5720 | 19900 | 0.0           | -               |
| 6.5885 | 19950 | 0.0           | -               |
| 6.6050 | 20000 | 0.0           | -               |
| 6.6215 | 20050 | 0.0           | -               |
| 6.6380 | 20100 | 0.0           | -               |
| 6.6546 | 20150 | 0.0           | -               |
| 6.6711 | 20200 | 0.0           | -               |
| 6.6876 | 20250 | 0.0           | -               |
| 6.7041 | 20300 | 0.0           | -               |
| 6.7206 | 20350 | 0.0           | -               |
| 6.7371 | 20400 | 0.0           | -               |
| 6.7536 | 20450 | 0.0           | -               |
| 6.7701 | 20500 | 0.0           | -               |
| 6.7867 | 20550 | 0.0           | -               |
| 6.8032 | 20600 | 0.0           | -               |
| 6.8197 | 20650 | 0.0           | -               |
| 6.8362 | 20700 | 0.0           | -               |
| 6.8527 | 20750 | 0.0           | -               |
| 6.8692 | 20800 | 0.0           | -               |
| 6.8857 | 20850 | 0.0           | -               |
| 6.9022 | 20900 | 0.0           | -               |
| 6.9188 | 20950 | 0.0           | -               |
| 6.9353 | 21000 | 0.0           | -               |
| 6.9518 | 21050 | 0.0           | -               |
| 6.9683 | 21100 | 0.0           | -               |
| 6.9848 | 21150 | 0.0           | -               |

### Framework Versions
- Python: 3.10.13
- SetFit: 1.0.3
- Sentence Transformers: 3.0.1
- spaCy: 3.4.2
- Transformers: 4.36.2
- PyTorch: 2.1.2
- Datasets: 2.20.0
- Tokenizers: 0.15.2

## Citation

### BibTeX
```bibtex
@article{https://doi.org/10.48550/arxiv.2209.11055,
    doi = {10.48550/ARXIV.2209.11055},
    url = {https://arxiv.org/abs/2209.11055},
    author = {Tunstall, Lewis and Reimers, Nils and Jo, Unso Eun Seo and Bates, Luke and Korat, Daniel and Wasserblat, Moshe and Pereg, Oren},
    keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {Efficient Few-Shot Learning Without Prompts},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->