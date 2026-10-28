# MNIST ile MLP rakam tanıma

Bu projede MNIST veri setindeki 28×28 gri tonlamalı rakamları, tamamen bağlantılı katmanlardan oluşan bir MLP ile sınıflandırıyorum. CNN kullanmıyorum. Üç farklı gizli katman genişliğini aynı veri bölünmesi ve eğitim ayarlarıyla karşılaştırıyorum.

## Proje yapısı

```text
mnist_mlp_assignment/
├── main.py                    # Deneyleri sırayla çalıştıran başlangıç dosyası
├── mnist_mlp/
│   ├── config.py              # Hiperparametreler ve model denemeleri
│   ├── data.py                # MNIST indirme ve veri kümelerine ayırma
│   ├── model.py               # MLP mimarisi
│   ├── training.py            # Eğitim döngüsü ve ölçümler
│   ├── visualization.py       # Loss ve accuracy grafikleri
│   └── handwriting.py         # El yazısı görsellerini hazırlama/tahmin
├── handwriting/               # Öğrencinin kendi rakam görselleri
├── requirements.txt
└── README.md
```

## Kurulum ve çalıştırma

Python 3.10 veya üzeri önerilir.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

İlk çalıştırmada MNIST otomatik indirilir. Eğitim çıktıları projenin `results` klasörüne kaydedilir. Program veri, sonuç ve `handwriting` klasörlerini proje dosyasının konumuna göre bulur; bu nedenle terminalin hangi klasörde olduğu fark etmez. Eğitim süresi bilgisayardaki işlemci/GPU'ya bağlıdır.

## Veri bölünmesi ve yöntem

- Eğitim: 55.000 görüntü
- Doğrulama: 5.000 görüntü
- Test: MNIST'in ayrı 10.000 görüntüsü
- Girdi: 28×28 görüntü; `ToTensor` ile piksel değerleri [0, 1] aralığına çevrilip 784 elemana açılır.
- Kayıp: CrossEntropyLoss
- Optimizasyon: Adam, öğrenme oranı 0.001
- Batch size: 128
- Epoch: 10
- Aktivasyon: ReLU
- Rastgelelik tohumu: 42

| Deney | Gizli katmanlar |
| --- | --- |
| mlp_128_64 | 128, 64 |
| mlp_256_128 | 256, 128 |
| mlp_512_256 | 512, 256 |

Her epoch sonunda eğitim ve doğrulama loss/accuracy değerleri hesaplanır. Son epoch doğrulama accuracy'si en yüksek olan yapı seçilir. Test kümesi model seçerken kullanılmaz; seçilen modele yalnızca son değerlendirmede uygulanır.

## Üretilen sonuçlar

- `results/model_comparison.csv`: modellerin loss, accuracy ve eğitim süreleri; seçilen modelin test ölçümü
- `results/metrics.json`: veri boyutları, seçilen model ve deney ölçümleri
- `results/training_curves.png`: seçilen modelin eğitim/doğrulama loss ve accuracy grafikleri
- `results/best_mlp.pt`: validation sonucuna göre seçilen model ağırlıkları
- `results/handwriting_predictions.csv`: eklenen el yazısı görsellerinin tahminleri

## Kendi el yazım ile deneme

Öğrenci numaramdaki rakamları kendim yazıp her rakamı ayrı görsel olarak `handwriting` klasörüne koyuyorum. Örneğin dosyaları `rakam_1.png`, `rakam_2.png` şeklinde adlandırabilirim. Her resimde tek rakam bulunmalıdır. PNG, JPG ve BMP desteklenir.

Açık renkli kağıda koyu kalemle yazıp rakamı kadrajda büyükçe tutmak iyi sonuç verir. Kod arka planı köşe piksellerinden anlamaya çalışır, rakamı kırpar ve 28×28 siyah zemin/beyaz rakam biçimine getirir. El yazısı MNIST'ten farklı olabileceği için yanlış tahminler normaldir; bu farkı raporda tartışıyorum.

## Sonuç değerlendirmesi

Üç MLP yapısını aynı veri bölünmesi ve eğitim ayarlarıyla karşılaştırdım. Validation sonucuna göre `mlp_128_64` modelini seçtim. Bu modelin validation accuracy değeri %97,68 oldu. Diğer iki modelin validation sonuçları %97,48 ve %97,44 çıktı. Daha büyük modelin eğitim verisindeki başarısı daha yüksek olsa da validation başarısı artmadı.

Seçilen modelin son epoch eğitim loss değeri 0,0224, validation loss değeri 0,0853 oldu. Eğitim accuracy değeri %99,35, validation accuracy değeri %97,68 olarak ölçüldü. Eğitim ve doğrulama arasındaki fark, modelin eğitim örneklerini biraz daha iyi öğrendiğini gösteriyor. Seçilen model MNIST test kümesinde %97,68 başarı elde etti. Eğitim yaklaşık 116 saniye sürdü.

Kendi yazdığım altı rakamı da ayrı dosyalar halinde modele verdim. Görsellerdeki gerçek rakamları kendim 2, 5, 0, 1, 3 ve 8 olarak belirledim. Sonuçlar şöyle oldu:

| Dosya | Gerçek rakam | Model tahmini | Güven |
| --- | ---: | ---: | ---: |
| 01.jpg | 2 | 0 | %99,84 |
| 02.jpg | 5 | 7 | %48,72 |
| 03.jpg | 0 | 2 | %34,61 |
| 04.jpg | 1 | 1 | %26,80 |
| 05.jpg | 3 | 9 | %22,74 |
| 06.jpg | 8 | 3 | %43,69 |

Bu altı görselden yalnızca `04.jpg` doğru tahmin edildi. Özellikle `01.jpg` için model yanlış tahmin yapmasına rağmen güven değeri çok yüksek çıktı. Bu yüzden güven değerini tahminin kesin doğruluğu olarak yorumlamamak gerektiğini gördüm.

Sonuçları incelerken görüntü hazırlama adımında da bir sorun fark ettim. Fotoğraflar siyah zemin/beyaz rakam biçimine çevrilip küçültülürken kâğıttaki bazı gölgeler de rakam gibi kalmış. Örneğin 01 ve 06 numaralı görsellerin 28×28 önizlemesinde fazladan beyaz bölgeler görünüyor. Bu durum rakamın kırpılmasını ve modelin gördüğü şekli bozmuş olabilir. Ayrıca kendi yazım biçimim MNIST örneklerinden farklı olduğu için modelin bu görüntülerde daha düşük başarı göstermesi beklenebilir.

Bence bu deneyde MNIST test başarısının yüksek olması, modelin benim el yazımı da aynı şekilde tanıyacağı anlamına gelmedi. Sonraki denemede daha aydınlık ve gölgesiz fotoğraf çekip ön işleme adımını iyileştirmeyi, sonra tahminleri tekrar karşılaştırmayı planlıyorum.
