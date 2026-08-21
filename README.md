# YZ50 — Neural Networks, 1. Hafta

Bu depo, sinir ağlarının temel matematiğini herhangi bir üçüncü taraf
kütüphane kullanmadan gösteren beş küçük çalışmayı içerir:

1. Tek nöron forward pass
2. Birden fazla nörondan oluşan katman
3. Mean squared error (MSE) loss
4. Bir parametreyi manuel değiştirerek loss'u gözlemleme ve eğri çizme
5. Sayısal türev ile gradient descent

## Çalıştırma

Python 3 ile:

```bash
python main.py
```

Program terminalde bütün adımları gösterir ve `outputs/` altında şunları üretir:

- `loss_curve.svg`: weight değiştikçe loss eğrisi
- `gradient_descent.csv`: her gradient descent adımındaki weight ve loss

Testler:

```bash
python -m unittest -v
```

## Dosyalar

- `neural_network.py`: Matematiksel fonksiyonlar
- `main.py`: Beş görevi sırayla çalıştıran deney
- `test_neural_network.py`: Fonksiyonların doğrulama testleri

Örnekteki veri `y = 2x` ilişkisidir. Modelin tek ağırlığı başlangıçta `-1`
olduğu için tahminleri kötüdür. Gradient descent, loss'un ağırlığa göre sayısal
türevini hesaplayıp ağırlığı ters yönde günceller; ağırlık giderek `2` değerine
yaklaşırken loss sıfıra yaklaşır.
