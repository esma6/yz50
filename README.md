# YZ50 — Neural Networks, 1. Hafta Çalışma Defteri

Bu proje, sinir ağlarının temelinde gerçekleşen işlemleri hazır makine öğrenmesi
kütüphaneleri kullanmadan anlamak için hazırlandı. Kod yalnızca Python standart
kütüphanesini kullanıyor. Böylece bir framework'ün arkasına saklanan çarpma,
toplama, hata hesaplama, türev alma ve parametre güncelleme adımlarını doğrudan
görebiliyorum.

## Temel kavramlar

- **Nöron:** Girdileri ağırlıklarla çarpıp toplar ve sonuca bias ekler.
- **Ağırlık (weight):** Bir girdinin sonuç üzerindeki etkisini belirleyen,
  modelin öğrenme sırasında değiştirdiği parametredir.
- **Bias:** Nöronun çıktısını sabit miktarda kaydıran, öğrenilebilir parametredir.
- **Katman (layer):** Aynı girdileri alan birden fazla nöronun oluşturduğu yapıdır.
- **Forward pass:** Girdiden başlayıp tahmine ulaşana kadar yapılan hesaplamadır.
- **Loss:** Model tahminiyle gerçek değer arasındaki hatayı tek bir sayıyla ölçer.
- **Gradient:** Bir parametre değiştiğinde loss'un hangi yönde ve ne hızda
  değiştiğini gösterir.
- **Öğrenme:** Loss'u azaltacak ağırlık ve bias değerlerinin bulunmasıdır.

## 1. Tek nöron ve forward pass

Bir nöronun en temel hali doğrusal bir fonksiyondur:

```text
çıktı = x₁w₁ + x₂w₂ + ... + xₙwₙ + bias
```

Burada `x` değerleri girdileri, `w` değerleri ağırlıkları temsil eder. Forward
pass sırasında her girdi kendi ağırlığıyla çarpılır, sonuçlar toplanır ve bias
eklenir. Bu aşamada model henüz öğrenmez; yalnızca mevcut parametreleriyle bir
çıktı hesaplar.

Projede `neuron_forward()` fonksiyonu şu örneği çalıştırır:

```text
inputs  = [1.0, 2.0, 3.0]
weights = [0.2, 0.8, -0.5]
bias    = 2.0

çıktı = (1 × 0.2) + (2 × 0.8) + (3 × -0.5) + 2
      = 0.2 + 1.6 - 1.5 + 2
      = 2.3
```

Ağırlıkların işareti de önemlidir: pozitif ağırlık girdinin çıktıyı artırmasına,
negatif ağırlık azaltmasına neden olabilir. Bias ise tüm girdiler sıfır olsa
bile nöronun sıfırdan farklı çıktı üretebilmesini sağlar.

## 2. Birden fazla nörondan oluşan katman

Bir katmanda birden fazla nöron bulunur. Bu projedeki her nöron aynı üç girdiyi
alır; fakat kendi ağırlık ve bias değerlerine sahip olduğu için farklı bir çıktı
üretir:

```text
aynı inputs ──┬──> nöron 1 ──> 2.30
              ├──> nöron 2 ──> 2.46
              └──> nöron 3 ──> 0.21
```

`layer_forward()` her nöron için `neuron_forward()` fonksiyonunu çağırır ve
sonuçları bir liste halinde döndürür. Üç nöronlu katmanın çıktısı bu nedenle üç
sayıdan oluşur: `[2.3, 2.46, 0.21]`.

Bir katmanın çıktıları sonraki katmanın girdileri yapılabilir. Sinir ağlarının
katmanlı yapısı bu fikrin tekrar edilmesinden doğar. Bu çalışmada aktivasyon
fonksiyonu eklenmediği için nöronlar yalnızca doğrusal hesaplama yapmaktadır.

## 3. Loss fonksiyonu: Mean Squared Error

Forward pass bir tahmin üretir, fakat tahminin ne kadar iyi olduğunu tek başına
söylemez. Bunun için tahminlerle hedefler arasındaki farkı ölçen bir loss
fonksiyonu gerekir. Bu projede Mean Squared Error (ortalama karesel hata)
kullanılır:

```text
MSE = [(tahmin₁ - hedef₁)² + ... + (tahminₙ - hedefₙ)²] / n
```

Örnekte:

```text
tahminler = [2.5, 0.0, 2.1]
hedefler  = [3.0, -0.5, 2.0]

kare hatalar = [(-0.5)², (0.5)², (0.1)²]
              = [0.25, 0.25, 0.01]
MSE           = 0.51 / 3 = 0.17
```

Farkların karesini almak negatif ve pozitif hataların birbirini götürmesini
engeller. Ayrıca büyük hataları daha fazla cezalandırır. Loss'un küçük olması
tahminlerin hedeflere yakın olduğunu gösterir; sıfır loss bu veri üzerinde
kusursuz tahmin demektir.

## 4. Parametreyi manuel değiştirmek ve loss eğrisi

Modelin öğrenmesinin ne anlama geldiğini görmek için basit bir veri kümesi
kullanılır:

```text
x       = [-2, -1, 0, 1, 2]
hedef y = [-4, -2, 0, 2, 4]
```

Verinin gerçek ilişkisi `y = 2x` şeklindedir. Model ise `tahmin = weight × x +
bias` formülünü kullanır ve bu deneyde bias sıfırda sabit tutulur. Weight değeri
`-1.0` ile `3.0` arasında elle değiştirilerek her değer için bütün veri
kümesinin MSE loss'u hesaplanır.

| Weight | Loss | Yorum |
|------:|-----:|---|
| -1.0 | 18.00 | Tahminler gerçek ilişkinin ters yönünde |
| 0.0 | 8.00 | Model bütün girdiler için sıfır üretiyor |
| 1.0 | 2.00 | Doğru yönde, fakat eğim yetersiz |
| 2.0 | 0.00 | Gerçek ilişki bulundu |
| 3.0 | 2.00 | Doğru yönde, fakat eğim fazla |

Bu değerler `outputs/loss_curve.svg` dosyasında çizilir. Eğrinin en düşük
noktası `weight = 2` değerindedir. Dolayısıyla öğrenme, loss eğrisinin en düşük
noktasına karşılık gelen parametreyi arama süreci olarak düşünülebilir.

## 5. Sayısal türev ve gradient descent

Weight değerlerini elle denemek küçük bir örnekte mümkündür; gerçek modellerde
milyonlarca veya milyarlarca parametre olabilir. Parametreyi hangi yönde
değiştirmemiz gerektiğini türev yardımıyla buluruz.

Bu projede türev, merkezi fark yöntemiyle sayısal olarak yaklaşık hesaplanır:

```text
f'(w) ≈ [f(w + ε) - f(w - ε)] / (2ε)
```

Burada `f(w)` belirli bir weight değerindeki loss, `ε` ise çok küçük bir
sayıdır (`1e-5`). Weight'in biraz sağındaki ve biraz solundaki loss değerlerini
karşılaştırmak eğimin yönünü verir:

- Gradient pozitifse weight arttıkça loss artıyordur; weight azaltılmalıdır.
- Gradient negatifse weight arttıkça loss azalıyordur; weight artırılmalıdır.
- Gradient sıfıra yakınsa eğrinin düzleştiği bir noktaya yaklaşılmıştır.

Gradient descent güncelleme kuralı şöyledir:

```text
yeni_weight = eski_weight - learning_rate × gradient
```

`learning_rate` adım büyüklüğüdür. Çok büyük seçilirse minimum nokta aşılabilir
ve süreç kararsızlaşabilir; çok küçük seçilirse öğrenme gereğinden yavaş olur.
Bu deneyde learning rate `0.1`, başlangıç weight'i `-1.0` ve adım sayısı `20`
olarak seçildi.

| Adım | Weight | Loss |
|----:|-------:|-----:|
| 0 | -1.000000 | 18.00000000 |
| 1 | 0.200000 | 6.48000000 |
| 5 | 1.766720 | 0.10883912 |
| 10 | 1.981860 | 0.00065811 |
| 20 | 1.999890 | 0.00000002 |

Her güncellemede weight `2` değerine yaklaşırken loss küçülür. Modelin
“öğrenmesi” burada yeni bilgi ezberlemesi değil, veriyle daha uyumlu parametre
değerlerine ulaşmasıdır. Bütün adımlar `outputs/gradient_descent.csv` dosyasına
kaydedilir.

## Language model ile bağlantısı

Bu örnek tek bir sayıyı tahmin ediyor; bir language model ise önceki token'ları
girdi olarak alıp sonraki token için olasılıklar üretir. Ölçek çok daha büyük
olsa da temel eğitim döngüsü benzerdir:

1. Model mevcut parametreleriyle forward pass yapar.
2. Tahmin ile doğru hedef arasındaki loss hesaplanır.
3. Gradient'ler parametrelerin loss'a etkisini gösterir.
4. Parametreler loss'u azaltacak yönde güncellenir.
5. Bu süreç çok sayıda veri örneği üzerinde tekrar edilir.

Gerçek sinir ağlarında gradient'ler sayısal türev yerine çok daha verimli olan
backpropagation ve otomatik türev araçlarıyla hesaplanır.

## Projeyi çalıştırma

Python 3 yeterlidir; ek paket kurulumu gerekmez:

```bash
python main.py
```

Program beş deneyi terminalde sırayla gösterir ve `outputs/` klasörünü üretir.

Testleri çalıştırmak için:

```bash
python -m unittest -v
```

## Dosya yapısı

| Dosya | Görevi |
|---|---|
| `neural_network.py` | Nöron, katman, loss, sayısal türev ve gradient descent fonksiyonları |
| `main.py` | Beş çalışmayı sırayla çalıştırır, sonuçları yazdırır ve çıktı dosyalarını üretir |
| `test_neural_network.py` | Matematiksel fonksiyonların beklenen sonuçları verdiğini doğrular |
| `outputs/loss_curve.svg` | Weight değişimine karşı loss eğrisi |
| `outputs/gradient_descent.csv` | Her optimizasyon adımındaki weight ve loss değerleri |

## Bu çalışmadan çıkardığım sonuçlar

- Nöronun yaptığı temel işlem ağırlıklı toplam ve bias eklemektir.
- Forward pass tahmin üretir; tek başına öğrenme gerçekleştirmez.
- Loss fonksiyonu modelin hatasını ölçülebilir bir hedefe dönüştürür.
- Parametre değiştiğinde loss da değişir; eğitim doğru parametreleri aramaktır.
- Türev loss'un artış yönünü, negatif gradient yönü ise azalış yönünü gösterir.
- Gradient descent küçük ve tekrarlı güncellemelerle loss'u düşürür.
- Büyük sinir ağları da daha fazla katman ve parametreyle aynı temel döngüyü
  uygular.
