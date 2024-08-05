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
- text: Turkcell:Turkcell sürekli 1 ayda 3 fatura kesiyor. Haksız yere faturalıdan
    faturasız hatta geçtim. Ceza adı altında bir ayda 3 fatura kesmesi mantık dışı
    bir hareket. Bu resmen dalga geçmek. Gelen faturaları ödemeyeceğim ve ödediğim
    paraların iadesini istiyorum. Sorunum çözülmezse mahkemeye konuyu taşıyacağım.
    Onunla da kalmayacağım, TV programlarına konuyu taşıyacağım. Artık haberlerde
    markanızı görürsünüz. Bir değil, iki değil, bu ne böyle?
- text: Turkcell:Turkcell art niyetli olmuş. Daha 10-15 gün geçmeden faturayı ödemediyseniz
    160 TL açma kapama ücreti yansıtılıyor. Normalde 2. Fatura geldiğinde 1. Ödenmemişse
    yansıtılır. Büyük yanıltma yapıyor. Müşteri hizmetleri yapacak bir şey yok diyor.
    Artık Turkcell hayatımdan çıkarıyorum.
- text: Türk Telekom:Türk Telekom hiçbir şekilde internet ve şebeke çekmiyor. Çok
    zorluk çekiliyor, gerekli çalışmaları da yapmıyorlar. Birkaç defa arayıp bildirdim
    ama hiçbir geri dönüş yok. Kocaeli Çayırova'dan çevreme de sordum, aynı cevapları
    aldım. Hiçbir şekilde çekmiyor.
- text: Mavi Jeans:Mavi Jeans yeni sezon kot pantolonları hem şık hem de rahat. Esnek
    kumaşı sayesinde gün boyu konfor sağlıyor. Renk seçenekleri de oldukça zengin.
- text: Starbucks:Geçen hafta sonu Starbucks arkadaşlarımla buluştuk ve kahvelerinin
    her zamanki gibi taze ve lezzetli olduğunu fark ettim. Çalışanlar da her zamanki
    gibi güler yüzlü ve yardımseverdi.
inference: false
---

# SetFit Aspect Model with loodos/bert-base-turkish-uncased

This is a [SetFit](https://github.com/huggingface/setfit) model that can be used for Aspect Based Sentiment Analysis (ABSA). This SetFit model uses [loodos/bert-base-turkish-uncased](https://huggingface.co/loodos/bert-base-turkish-uncased) as the Sentence Transformer embedding model. A [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance is used for classification. In particular, this model is in charge of filtering aspect span candidates.

The model has been trained using an efficient few-shot learning technique that involves:

1. Fine-tuning a [Sentence Transformer](https://www.sbert.net) with contrastive learning.
2. Training a classification head with features from the fine-tuned Sentence Transformer.

This model was trained within the context of a larger system for ABSA, which looks like so:

1. Use a spaCy model to select possible aspect span candidates.
2. **Use this SetFit model to filter these possible aspect span candidates.**
3. Use a SetFit model to classify the filtered aspect span candidates.

## Model Details

### Model Description
- **Model Type:** SetFit
- **Sentence Transformer body:** [loodos/bert-base-turkish-uncased](https://huggingface.co/loodos/bert-base-turkish-uncased)
- **Classification head:** a [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance
- **spaCy Model:** tr_core_news_trf_post_process
- **SetFitABSA Aspect Model:** [tddi-aspect-model](https://huggingface.co/tddi-aspect-model)
- **SetFitABSA Polarity Model:** [setfit-absa-polarity](https://huggingface.co/setfit-absa-polarity)
- **Maximum Sequence Length:** 512 tokens
- **Number of Classes:** 2 classes
<!-- - **Training Dataset:** [Unknown](https://huggingface.co/datasets/unknown) -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Repository:** [SetFit on GitHub](https://github.com/huggingface/setfit)
- **Paper:** [Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055)
- **Blogpost:** [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)

### Model Labels
| Label     | Examples                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
|:----------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| aspect    | <ul><li>'sac serum markası:kullanmadığım sac serum markası kalmadi diyebilirim ...Ancak bioxin gerçekten çok başarılı saçlarım uzun ve gür agirlastirmadan yumuşacık yapıyor...sadece kokusu çok iyi değil umarım onun içinde bir formül bulunur .... stok yapılabilecek şekilde başarılı fiyatı gramajina göre çok iyi bence...'</li><li>'bioxin:kullanmadığım sac serum markası kalmadi diyebilirim ...Ancak bioxin gerçekten çok başarılı saçlarım uzun ve gür agirlastirmadan yumuşacık yapıyor...sadece kokusu çok iyi değil umarım onun içinde bir formül bulunur .... stok yapılabilecek şekilde başarılı fiyatı gramajina göre çok iyi bence...'</li><li>'Loreal bioblas:Loreal bioblas yağlarını kullanıyordum benim saçlarım kuru ve sarıya boyatıyorum sürekli saçıma normalde bayağı sürmem gerekiyor kuru olduğu için ama çok az sürmeme rağmen nemlendirdi parlattı yumuşacık yaptı yağlı saçlar sanırım 1 yılda anca bitirir bu yağı ben çok beğendim kuru saçlar öncelikli almalı bence'</li></ul>                                                                                                                                   |
| no aspect | <ul><li>'Popeyes:TAB Gıda bağlı\xa0Popeyes\xa0markalı İzmir Optimum AVM bulunan restoranda sipariş verdim. Siparişimi kasa arkasında bulunan ve herkese yayınlanmakta olan büyük ekranlı dijital menüden fiyatı 190 TL olarak gösterilen bir sandviç menü olarak seçtim. Personel siparişimi alıp kasasında\xa0işlediğinde ise 200 TL olarak bir ödeme istedi. Konuyu Tüket Hakem Heyeti ve CİMER de taşıyacağım.'</li><li>'İzmir Optimum AVM:TAB Gıda bağlı\xa0Popeyes\xa0markalı İzmir Optimum AVM bulunan restoranda sipariş verdim. Siparişimi kasa arkasında bulunan ve herkese yayınlanmakta olan büyük ekranlı dijital menüden fiyatı 190 TL olarak gösterilen bir sandviç menü olarak seçtim. Personel siparişimi alıp kasasında\xa0işlediğinde ise 200 TL olarak bir ödeme istedi. Konuyu Tüket Hakem Heyeti ve CİMER de taşıyacağım.'</li><li>'EasyCep:Samsung cihazı ile değiştirme kampanyası için ürün fiyatı harici ek 5000 TL kampanya desteğimin ödenmemesi sorununu da Amazon atıyorlar. EasyCep resmen oyalıyor ödememek için. Ücreti Tüket Hakem Heyet bildireceğim. 40 günü aşkın süre olmasına rağmen hala ödenmedi.'</li></ul> |

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
    "tddi-aspect-model",
    "setfit-absa-polarity",
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
| Word count   | 8   | 39.2150 | 96  |

| Label     | Training Sample Count |
|:----------|:----------------------|
| no aspect | 105                   |
| aspect    | 537                   |

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
| 0.0004 | 1     | 0.3773        | -               |
| 0.0177 | 50    | 0.3563        | -               |
| 0.0355 | 100   | 0.3062        | -               |
| 0.0532 | 150   | 0.2641        | -               |
| 0.0709 | 200   | 0.2246        | -               |
| 0.0887 | 250   | 0.2558        | -               |
| 0.1064 | 300   | 0.2195        | -               |
| 0.1241 | 350   | 0.1885        | -               |
| 0.1418 | 400   | 0.1468        | -               |
| 0.1596 | 450   | 0.1216        | -               |
| 0.1773 | 500   | 0.19          | -               |
| 0.1950 | 550   | 0.0739        | -               |
| 0.2128 | 600   | 0.1124        | -               |
| 0.2305 | 650   | 0.1101        | -               |
| 0.2482 | 700   | 0.1034        | -               |
| 0.2660 | 750   | 0.0994        | -               |
| 0.2837 | 800   | 0.0865        | -               |
| 0.3014 | 850   | 0.0562        | -               |
| 0.3191 | 900   | 0.0706        | -               |
| 0.3369 | 950   | 0.0396        | -               |
| 0.3546 | 1000  | 0.0728        | -               |
| 0.3723 | 1050  | 0.0229        | -               |
| 0.3901 | 1100  | 0.0509        | -               |
| 0.4078 | 1150  | 0.0749        | -               |
| 0.4255 | 1200  | 0.0874        | -               |
| 0.4433 | 1250  | 0.1094        | -               |
| 0.4610 | 1300  | 0.0367        | -               |
| 0.4787 | 1350  | 0.0499        | -               |
| 0.4965 | 1400  | 0.0379        | -               |
| 0.5142 | 1450  | 0.0602        | -               |
| 0.5319 | 1500  | 0.093         | -               |
| 0.5496 | 1550  | 0.0811        | -               |
| 0.5674 | 1600  | 0.0395        | -               |
| 0.5851 | 1650  | 0.0447        | -               |
| 0.6028 | 1700  | 0.1141        | -               |
| 0.6206 | 1750  | 0.0397        | -               |
| 0.6383 | 1800  | 0.0885        | -               |
| 0.6560 | 1850  | 0.0862        | -               |
| 0.6738 | 1900  | 0.0814        | -               |
| 0.6915 | 1950  | 0.0683        | -               |
| 0.7092 | 2000  | 0.0422        | -               |
| 0.7270 | 2050  | 0.0581        | -               |
| 0.7447 | 2100  | 0.1225        | -               |
| 0.7624 | 2150  | 0.0929        | -               |
| 0.7801 | 2200  | 0.0798        | -               |
| 0.7979 | 2250  | 0.096         | -               |
| 0.8156 | 2300  | 0.0848        | -               |
| 0.8333 | 2350  | 0.081         | -               |
| 0.8511 | 2400  | 0.0217        | -               |
| 0.8688 | 2450  | 0.0651        | -               |
| 0.8865 | 2500  | 0.0635        | -               |
| 0.9043 | 2550  | 0.0696        | -               |
| 0.9220 | 2600  | 0.0837        | -               |
| 0.9397 | 2650  | 0.0779        | -               |
| 0.9574 | 2700  | 0.067         | -               |
| 0.9752 | 2750  | 0.0518        | -               |
| 0.9929 | 2800  | 0.0425        | -               |
| 1.0106 | 2850  | 0.0589        | -               |
| 1.0284 | 2900  | 0.056         | -               |
| 1.0461 | 2950  | 0.0525        | -               |
| 1.0638 | 3000  | 0.0567        | -               |
| 1.0816 | 3050  | 0.0765        | -               |
| 1.0993 | 3100  | 0.1157        | -               |
| 1.1170 | 3150  | 0.0632        | -               |
| 1.1348 | 3200  | 0.094         | -               |
| 1.1525 | 3250  | 0.0439        | -               |
| 1.1702 | 3300  | 0.0321        | -               |
| 1.1879 | 3350  | 0.1041        | -               |
| 1.2057 | 3400  | 0.0543        | -               |
| 1.2234 | 3450  | 0.0804        | -               |
| 1.2411 | 3500  | 0.0541        | -               |
| 1.2589 | 3550  | 0.0416        | -               |
| 1.2766 | 3600  | 0.0691        | -               |
| 1.2943 | 3650  | 0.0334        | -               |
| 1.3121 | 3700  | 0.0557        | -               |
| 1.3298 | 3750  | 0.0595        | -               |
| 1.3475 | 3800  | 0.0557        | -               |
| 1.3652 | 3850  | 0.0592        | -               |
| 1.3830 | 3900  | 0.1116        | -               |
| 1.4007 | 3950  | 0.0625        | -               |
| 1.4184 | 4000  | 0.0533        | -               |
| 1.4362 | 4050  | 0.0188        | -               |
| 1.4539 | 4100  | 0.0417        | -               |
| 1.4716 | 4150  | 0.0479        | -               |
| 1.4894 | 4200  | 0.0305        | -               |
| 1.5071 | 4250  | 0.0308        | -               |
| 1.5248 | 4300  | 0.0662        | -               |
| 1.5426 | 4350  | 0.0728        | -               |
| 1.5603 | 4400  | 0.0294        | -               |
| 1.5780 | 4450  | 0.1188        | -               |
| 1.5957 | 4500  | 0.0558        | -               |
| 1.6135 | 4550  | 0.0568        | -               |
| 1.6312 | 4600  | 0.0412        | -               |
| 1.6489 | 4650  | 0.0669        | -               |
| 1.6667 | 4700  | 0.0684        | -               |
| 1.6844 | 4750  | 0.0445        | -               |
| 1.7021 | 4800  | 0.055         | -               |
| 1.7199 | 4850  | 0.071         | -               |
| 1.7376 | 4900  | 0.0704        | -               |
| 1.7553 | 4950  | 0.0407        | -               |
| 1.7730 | 5000  | 0.089         | -               |
| 1.7908 | 5050  | 0.0651        | -               |
| 1.8085 | 5100  | 0.1039        | -               |
| 1.8262 | 5150  | 0.0758        | -               |
| 1.8440 | 5200  | 0.0367        | -               |
| 1.8617 | 5250  | 0.0602        | -               |
| 1.8794 | 5300  | 0.0397        | -               |
| 1.8972 | 5350  | 0.091         | -               |
| 1.9149 | 5400  | 0.0584        | -               |
| 1.9326 | 5450  | 0.0588        | -               |
| 1.9504 | 5500  | 0.0689        | -               |
| 1.9681 | 5550  | 0.0404        | -               |
| 1.9858 | 5600  | 0.0838        | -               |
| 2.0035 | 5650  | 0.0776        | -               |
| 2.0213 | 5700  | 0.0966        | -               |
| 2.0390 | 5750  | 0.0657        | -               |
| 2.0567 | 5800  | 0.093         | -               |
| 2.0745 | 5850  | 0.0659        | -               |
| 2.0922 | 5900  | 0.0921        | -               |
| 2.1099 | 5950  | 0.0591        | -               |
| 2.1277 | 6000  | 0.0823        | -               |
| 2.1454 | 6050  | 0.0473        | -               |
| 2.1631 | 6100  | 0.0134        | -               |
| 2.1809 | 6150  | 0.0829        | -               |
| 2.1986 | 6200  | 0.114         | -               |
| 2.2163 | 6250  | 0.0883        | -               |
| 2.2340 | 6300  | 0.0494        | -               |
| 2.2518 | 6350  | 0.0642        | -               |
| 2.2695 | 6400  | 0.0669        | -               |
| 2.2872 | 6450  | 0.0695        | -               |
| 2.3050 | 6500  | 0.0367        | -               |
| 2.3227 | 6550  | 0.0658        | -               |
| 2.3404 | 6600  | 0.0413        | -               |
| 2.3582 | 6650  | 0.0488        | -               |
| 2.3759 | 6700  | 0.0577        | -               |
| 2.3936 | 6750  | 0.1247        | -               |
| 2.4113 | 6800  | 0.0401        | -               |
| 2.4291 | 6850  | 0.0332        | -               |
| 2.4468 | 6900  | 0.0859        | -               |
| 2.4645 | 6950  | 0.0476        | -               |
| 2.4823 | 7000  | 0.0838        | -               |
| 2.5    | 7050  | 0.0563        | -               |
| 2.5177 | 7100  | 0.068         | -               |
| 2.5355 | 7150  | 0.0761        | -               |
| 2.5532 | 7200  | 0.0626        | -               |
| 2.5709 | 7250  | 0.0484        | -               |
| 2.5887 | 7300  | 0.0542        | -               |
| 2.6064 | 7350  | 0.0771        | -               |
| 2.6241 | 7400  | 0.0927        | -               |
| 2.6418 | 7450  | 0.0284        | -               |
| 2.6596 | 7500  | 0.0463        | -               |
| 2.6773 | 7550  | 0.0821        | -               |
| 2.6950 | 7600  | 0.0623        | -               |
| 2.7128 | 7650  | 0.0544        | -               |
| 2.7305 | 7700  | 0.0602        | -               |
| 2.7482 | 7750  | 0.0621        | -               |
| 2.7660 | 7800  | 0.0827        | -               |
| 2.7837 | 7850  | 0.0661        | -               |
| 2.8014 | 7900  | 0.0744        | -               |
| 2.8191 | 7950  | 0.0328        | -               |
| 2.8369 | 8000  | 0.0351        | -               |
| 2.8546 | 8050  | 0.0637        | -               |
| 2.8723 | 8100  | 0.0289        | -               |
| 2.8901 | 8150  | 0.0461        | -               |
| 2.9078 | 8200  | 0.0516        | -               |
| 2.9255 | 8250  | 0.0877        | -               |
| 2.9433 | 8300  | 0.0373        | -               |
| 2.9610 | 8350  | 0.0899        | -               |
| 2.9787 | 8400  | 0.0485        | -               |
| 2.9965 | 8450  | 0.0529        | -               |
| 3.0142 | 8500  | 0.0425        | -               |
| 3.0319 | 8550  | 0.0364        | -               |
| 3.0496 | 8600  | 0.0942        | -               |
| 3.0674 | 8650  | 0.048         | -               |
| 3.0851 | 8700  | 0.0781        | -               |
| 3.1028 | 8750  | 0.0406        | -               |
| 3.1206 | 8800  | 0.0928        | -               |
| 3.1383 | 8850  | 0.0435        | -               |
| 3.1560 | 8900  | 0.029         | -               |
| 3.1738 | 8950  | 0.0574        | -               |
| 3.1915 | 9000  | 0.0533        | -               |
| 3.2092 | 9050  | 0.0702        | -               |
| 3.2270 | 9100  | 0.0608        | -               |
| 3.2447 | 9150  | 0.054         | -               |
| 3.2624 | 9200  | 0.0468        | -               |
| 3.2801 | 9250  | 0.0431        | -               |
| 3.2979 | 9300  | 0.0554        | -               |
| 3.3156 | 9350  | 0.0802        | -               |
| 3.3333 | 9400  | 0.042         | -               |
| 3.3511 | 9450  | 0.0851        | -               |
| 3.3688 | 9500  | 0.0613        | -               |
| 3.3865 | 9550  | 0.0567        | -               |
| 3.4043 | 9600  | 0.0588        | -               |
| 3.4220 | 9650  | 0.065         | -               |
| 3.4397 | 9700  | 0.0514        | -               |
| 3.4574 | 9750  | 0.0265        | -               |
| 3.4752 | 9800  | 0.0432        | -               |
| 3.4929 | 9850  | 0.0224        | -               |
| 3.5106 | 9900  | 0.0818        | -               |
| 3.5284 | 9950  | 0.0705        | -               |
| 3.5461 | 10000 | 0.0496        | -               |
| 3.5638 | 10050 | 0.0794        | -               |
| 3.5816 | 10100 | 0.0607        | -               |
| 3.5993 | 10150 | 0.0502        | -               |
| 3.6170 | 10200 | 0.0704        | -               |
| 3.6348 | 10250 | 0.0531        | -               |
| 3.6525 | 10300 | 0.0282        | -               |
| 3.6702 | 10350 | 0.0622        | -               |
| 3.6879 | 10400 | 0.0432        | -               |
| 3.7057 | 10450 | 0.027         | -               |
| 3.7234 | 10500 | 0.096         | -               |
| 3.7411 | 10550 | 0.0709        | -               |
| 3.7589 | 10600 | 0.0782        | -               |
| 3.7766 | 10650 | 0.1034        | -               |
| 3.7943 | 10700 | 0.075         | -               |
| 3.8121 | 10750 | 0.0293        | -               |
| 3.8298 | 10800 | 0.0578        | -               |
| 3.8475 | 10850 | 0.1208        | -               |
| 3.8652 | 10900 | 0.0695        | -               |
| 3.8830 | 10950 | 0.0995        | -               |
| 3.9007 | 11000 | 0.0733        | -               |
| 3.9184 | 11050 | 0.0408        | -               |
| 3.9362 | 11100 | 0.0624        | -               |
| 3.9539 | 11150 | 0.0685        | -               |
| 3.9716 | 11200 | 0.0506        | -               |
| 3.9894 | 11250 | 0.0362        | -               |
| 4.0071 | 11300 | 0.0491        | -               |
| 4.0248 | 11350 | 0.0526        | -               |
| 4.0426 | 11400 | 0.057         | -               |
| 4.0603 | 11450 | 0.1162        | -               |
| 4.0780 | 11500 | 0.0487        | -               |
| 4.0957 | 11550 | 0.0744        | -               |
| 4.1135 | 11600 | 0.0563        | -               |
| 4.1312 | 11650 | 0.0718        | -               |
| 4.1489 | 11700 | 0.0667        | -               |
| 4.1667 | 11750 | 0.0508        | -               |
| 4.1844 | 11800 | 0.0598        | -               |
| 4.2021 | 11850 | 0.0914        | -               |
| 4.2199 | 11900 | 0.0627        | -               |
| 4.2376 | 11950 | 0.097         | -               |
| 4.2553 | 12000 | 0.0977        | -               |
| 4.2730 | 12050 | 0.059         | -               |
| 4.2908 | 12100 | 0.0368        | -               |
| 4.3085 | 12150 | 0.0453        | -               |
| 4.3262 | 12200 | 0.0621        | -               |
| 4.3440 | 12250 | 0.0542        | -               |
| 4.3617 | 12300 | 0.0735        | -               |
| 4.3794 | 12350 | 0.0673        | -               |
| 4.3972 | 12400 | 0.1024        | -               |
| 4.4149 | 12450 | 0.0414        | -               |
| 4.4326 | 12500 | 0.0805        | -               |
| 4.4504 | 12550 | 0.0741        | -               |
| 4.4681 | 12600 | 0.0566        | -               |
| 4.4858 | 12650 | 0.0856        | -               |
| 4.5035 | 12700 | 0.0642        | -               |
| 4.5213 | 12750 | 0.0474        | -               |
| 4.5390 | 12800 | 0.0374        | -               |
| 4.5567 | 12850 | 0.0954        | -               |
| 4.5745 | 12900 | 0.0763        | -               |
| 4.5922 | 12950 | 0.0925        | -               |
| 4.6099 | 13000 | 0.127         | -               |
| 4.6277 | 13050 | 0.0707        | -               |
| 4.6454 | 13100 | 0.0528        | -               |
| 4.6631 | 13150 | 0.0415        | -               |
| 4.6809 | 13200 | 0.0734        | -               |
| 4.6986 | 13250 | 0.0418        | -               |
| 4.7163 | 13300 | 0.0616        | -               |
| 4.7340 | 13350 | 0.0381        | -               |
| 4.7518 | 13400 | 0.0384        | -               |
| 4.7695 | 13450 | 0.0406        | -               |
| 4.7872 | 13500 | 0.0994        | -               |
| 4.8050 | 13550 | 0.0831        | -               |
| 4.8227 | 13600 | 0.0876        | -               |
| 4.8404 | 13650 | 0.1249        | -               |
| 4.8582 | 13700 | 0.0562        | -               |
| 4.8759 | 13750 | 0.0645        | -               |
| 4.8936 | 13800 | 0.0729        | -               |
| 4.9113 | 13850 | 0.0679        | -               |
| 4.9291 | 13900 | 0.0571        | -               |
| 4.9468 | 13950 | 0.0861        | -               |
| 4.9645 | 14000 | 0.0438        | -               |
| 4.9823 | 14050 | 0.0788        | -               |
| 5.0    | 14100 | 0.0785        | -               |
| 5.0177 | 14150 | 0.0194        | -               |
| 5.0355 | 14200 | 0.087         | -               |
| 5.0532 | 14250 | 0.0395        | -               |
| 5.0709 | 14300 | 0.0252        | -               |
| 5.0887 | 14350 | 0.0656        | -               |
| 5.1064 | 14400 | 0.0847        | -               |
| 5.1241 | 14450 | 0.0728        | -               |
| 5.1418 | 14500 | 0.0854        | -               |
| 5.1596 | 14550 | 0.0684        | -               |
| 5.1773 | 14600 | 0.0431        | -               |
| 5.1950 | 14650 | 0.0556        | -               |
| 5.2128 | 14700 | 0.0556        | -               |
| 5.2305 | 14750 | 0.0751        | -               |
| 5.2482 | 14800 | 0.0698        | -               |
| 5.2660 | 14850 | 0.0484        | -               |
| 5.2837 | 14900 | 0.0608        | -               |
| 5.3014 | 14950 | 0.0334        | -               |
| 5.3191 | 15000 | 0.0682        | -               |
| 5.3369 | 15050 | 0.089         | -               |
| 5.3546 | 15100 | 0.0864        | -               |
| 5.3723 | 15150 | 0.0682        | -               |
| 5.3901 | 15200 | 0.072         | -               |
| 5.4078 | 15250 | 0.0781        | -               |
| 5.4255 | 15300 | 0.0598        | -               |
| 5.4433 | 15350 | 0.0691        | -               |
| 5.4610 | 15400 | 0.0596        | -               |
| 5.4787 | 15450 | 0.0697        | -               |
| 5.4965 | 15500 | 0.0511        | -               |
| 5.5142 | 15550 | 0.0508        | -               |
| 5.5319 | 15600 | 0.069         | -               |
| 5.5496 | 15650 | 0.0726        | -               |
| 5.5674 | 15700 | 0.0412        | -               |
| 5.5851 | 15750 | 0.0546        | -               |
| 5.6028 | 15800 | 0.0854        | -               |
| 5.6206 | 15850 | 0.0539        | -               |
| 5.6383 | 15900 | 0.0935        | -               |
| 5.6560 | 15950 | 0.1026        | -               |
| 5.6738 | 16000 | 0.077         | -               |
| 5.6915 | 16050 | 0.0894        | -               |
| 5.7092 | 16100 | 0.0538        | -               |
| 5.7270 | 16150 | 0.0829        | -               |
| 5.7447 | 16200 | 0.0582        | -               |
| 5.7624 | 16250 | 0.1467        | -               |
| 5.7801 | 16300 | 0.0447        | -               |
| 5.7979 | 16350 | 0.078         | -               |
| 5.8156 | 16400 | 0.0564        | -               |
| 5.8333 | 16450 | 0.0717        | -               |
| 5.8511 | 16500 | 0.0459        | -               |
| 5.8688 | 16550 | 0.0732        | -               |
| 5.8865 | 16600 | 0.1024        | -               |
| 5.9043 | 16650 | 0.0416        | -               |
| 5.9220 | 16700 | 0.0653        | -               |
| 5.9397 | 16750 | 0.0549        | -               |
| 5.9574 | 16800 | 0.0376        | -               |
| 5.9752 | 16850 | 0.0923        | -               |
| 5.9929 | 16900 | 0.0859        | -               |
| 6.0106 | 16950 | 0.073         | -               |
| 6.0284 | 17000 | 0.0638        | -               |
| 6.0461 | 17050 | 0.0931        | -               |
| 6.0638 | 17100 | 0.0438        | -               |
| 6.0816 | 17150 | 0.0567        | -               |
| 6.0993 | 17200 | 0.0728        | -               |
| 6.1170 | 17250 | 0.1026        | -               |
| 6.1348 | 17300 | 0.0758        | -               |
| 6.1525 | 17350 | 0.0211        | -               |
| 6.1702 | 17400 | 0.0349        | -               |
| 6.1879 | 17450 | 0.0399        | -               |
| 6.2057 | 17500 | 0.0424        | -               |
| 6.2234 | 17550 | 0.0582        | -               |
| 6.2411 | 17600 | 0.0273        | -               |
| 6.2589 | 17650 | 0.0832        | -               |
| 6.2766 | 17700 | 0.0461        | -               |
| 6.2943 | 17750 | 0.0793        | -               |
| 6.3121 | 17800 | 0.0766        | -               |
| 6.3298 | 17850 | 0.0819        | -               |
| 6.3475 | 17900 | 0.078         | -               |
| 6.3652 | 17950 | 0.0614        | -               |
| 6.3830 | 18000 | 0.0626        | -               |
| 6.4007 | 18050 | 0.0788        | -               |
| 6.4184 | 18100 | 0.1125        | -               |
| 6.4362 | 18150 | 0.0305        | -               |
| 6.4539 | 18200 | 0.0603        | -               |
| 6.4716 | 18250 | 0.0247        | -               |
| 6.4894 | 18300 | 0.0552        | -               |
| 6.5071 | 18350 | 0.0298        | -               |
| 6.5248 | 18400 | 0.064         | -               |
| 6.5426 | 18450 | 0.0392        | -               |
| 6.5603 | 18500 | 0.0662        | -               |
| 6.5780 | 18550 | 0.0517        | -               |
| 6.5957 | 18600 | 0.0359        | -               |
| 6.6135 | 18650 | 0.0855        | -               |
| 6.6312 | 18700 | 0.0692        | -               |
| 6.6489 | 18750 | 0.0662        | -               |
| 6.6667 | 18800 | 0.0137        | -               |
| 6.6844 | 18850 | 0.0734        | -               |
| 6.7021 | 18900 | 0.0483        | -               |
| 6.7199 | 18950 | 0.0469        | -               |
| 6.7376 | 19000 | 0.0375        | -               |
| 6.7553 | 19050 | 0.0486        | -               |
| 6.7730 | 19100 | 0.0275        | -               |
| 6.7908 | 19150 | 0.052         | -               |
| 6.8085 | 19200 | 0.0836        | -               |
| 6.8262 | 19250 | 0.0307        | -               |
| 6.8440 | 19300 | 0.0787        | -               |
| 6.8617 | 19350 | 0.0423        | -               |
| 6.8794 | 19400 | 0.0521        | -               |
| 6.8972 | 19450 | 0.0395        | -               |
| 6.9149 | 19500 | 0.0553        | -               |
| 6.9326 | 19550 | 0.0594        | -               |
| 6.9504 | 19600 | 0.0809        | -               |
| 6.9681 | 19650 | 0.0412        | -               |
| 6.9858 | 19700 | 0.0449        | -               |

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