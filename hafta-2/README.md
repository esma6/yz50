# YZ50 — Backpropagation, 2. Hafta Çalışma Defteri

[← Ana sayfaya dön](../README.md)

Bu klasör, Karpathy'nin micrograd yaklaşımını izleyerek backpropagation
mekanizmasını sıfırdan anlamak için hazırlandı. Ana çalışma alanı
`backpropagation_calismasi.ipynb` notebook'udur. Açıklamalar, elle hesaplar,
deneyler ve çalıştırılmış çıktılar burada yan yana tutulur. Tamamlanan ve tekrar
kullanılacak sınıflar daha sonra temiz Python dosyalarına aktarılır.

## Bu haftanın ana fikri

Geçen hafta tek bir parametrenin türevini sayısal olarak hesapladım. Bu hafta,
büyük bir matematiksel ifadede bütün türevleri tek bir geriye geçişte hesaplayan
mekanizmayı kuracağım:

```text
Matematiksel işlemler
        ↓
Computation graph
        ↓
Her işlemin local derivative'i
        ↓
Chain rule ile geriye gradient akışı
        ↓
Bütün parametrelerin gradient'leri
```

## Dosya düzeni

| Dosya | Ne zaman doldurulacak? | Görevi |
|---|---|---|
| `backpropagation_calismasi.ipynb` | Bütün görevler | Ana çalışma defteri, deneyler ve görünür çıktılar |
| `value.py` | Görev 1–4 | `Value`, matematiksel operasyonlar ve `backward()` |
| `nn.py` | Görev 5 | `Neuron`, `Layer` ve `MLP` sınıfları |
| `train.py` | Görev 5 | Veri kümesi, loss ve eğitim döngüsü |
| `test_week2.py` | Bütün görevler | Analitik, sayısal ve davranışsal doğrulamalar |
| `build_notebook.py` | Gerektiğinde | Notebook'u temiz UTF-8 içerikle yeniden üretir |
| `requirements.txt` | Kurulum | Notebook, grafik ve PyTorch bağımlılıkları |

Notebook içinde bir kavramı deneyip doğruladıktan sonra kalıcı uygulamasını
ilgili `.py` dosyasına aktaracağım. Böylece notebook öğrenme sürecini, Python
dosyaları ise ortaya çıkan temiz uygulamayı gösterecek.

## Görev sırası ve bitirme ölçütleri

### 1. `Value` ve computation graph

Öğreneceğim kavramlar:

- Bir node'un `data` değeri
- Node'u üreten önceki node'lar
- Node'u meydana getiren operasyon
- Forward pass sırasında graph'ın oluşması

Yazacağım ilk parçalar:

1. Yalnızca scalar veri saklayan `Value`
2. Toplama işlemi
3. Çarpma işlemi
4. Önceki node ve operasyon bilgisinin saklanması

Bitirme ölçütü: Birkaç `Value` ile matematiksel ifade kurduğumda, her sonucun
hangi girdilerden ve hangi işlemden oluştuğunu inceleyebilmeliyim.

### 2. Gradient'leri elle hesaplamak

Önce basit bir ifadeyi, ardından tek nöronu elle geri geçeceğim. Bu aşamada
`backward()` henüz otomatik olmayacak.

Her node için şu iki soruyu cevaplayacağım:

1. Bu işlemin local derivative'i nedir?
2. Üstten gelen gradient ile local derivative chain rule'da nasıl birleşir?

Nöron örneğinde `tanh` operasyonunu ekleyeceğim.

Bitirme ölçütü: Elle yazdığım gradient değerlerini numerical derivative ile
doğrulayabilmeliyim.

### 3. Otomatik `backward()`

Kuracağım mekanizma:

1. Output node'un gradient'ini `1` yap.
2. Graph node'larını topological sıraya koy.
3. Bu sırayı tersine çevir.
4. Her node'un kendi local backward kuralını çalıştır.
5. Aynı değişkene birden fazla yoldan gelen gradient'leri topla.

Bitirme ölçütü: Elle hesapladığım basit ifade ve nöron gradient'leri,
`backward()` tarafından otomatik olarak aynı değerlerle üretilmeli.

### 4. Operasyonları parçalamak ve üçlü doğrulama

Eklenecek operasyonlar:

- `exp`
- Sabit kuvvet için `pow`
- Negatif alma ve çıkarma
- Bölme

`tanh`, daha küçük operasyonlardan yeniden oluşturulacak. Aynı ifadenin sonucu
ve gradient'leri üç yöntemle karşılaştırılacak:

```text
Kendi backward() sonucum
≈ numerical derivative sonucum
≈ PyTorch autograd sonucu
```

Bitirme ölçütü: Üç yöntemin değerleri küçük floating-point toleransı içinde
eşleşmeli.

### 5. `Neuron`, `Layer`, `MLP` ve eğitim

Kurulum sırası:

```text
Value → Neuron → Layer → MLP → Loss → Backward → Gradient descent
```

Modelin bütün weight ve bias değerleri tek parametre listesinde toplanacak.
Örnek küçük veri kümesinde eğitim yapılacak ve loss'un adım adım düştüğü
gösterilecek.

Her eğitim adımının değişmez sırası:

```text
Forward pass
→ Loss
→ Gradient'leri sıfırla
→ Backward pass
→ Parametreleri güncelle
```

Bitirme ölçütü: Loss kontrollü biçimde düşmeli ve tahminler hedeflere
yaklaşmalı. Her yeni backward öncesinde eski gradient'lerin sıfırlandığı ayrıca
test edilmeli.

## Her görev için çalışma yöntemi

Her kod parçasında şu sırayı uygulayacağım:

1. Önce kavramı kendi cümlelerimle açıklayacağım.
2. Küçük matematiksel örneği elle hesaplayacağım.
3. Yazacağım satırların sorumluluğunu belirleyeceğim.
4. Kodu kendim yazacağım.
5. Test veya numerical derivative ile doğrulayacağım.
6. Ne öğrendiğimi ve nerede zorlandığımı bu README'ye ekleyeceğim.

## Notebook çalışma düzeni

Her görev notebook içinde aynı dört hücre türüyle ilerler:

```text
Kavram — Kendi cümlelerimle açıklama
→ Elle hesap — Beklediğim sonuç
→ Kod deneyi — Yazdığım kodu çalıştırma
→ Sonuç — Çıktıyı yorumlama ve doğrulama
```

Notebook kaydedildiğinde hücre çıktıları dosyanın içinde kalır. GitHub bu
çıktıları doğrudan render ettiği için repoyu açan biri deneylerin sonuçlarını
kod çalıştırmadan görebilir.

## Çalıştırma

Dosyalar tamamlandıkça bu klasör içinde çalıştırılacak:

```bash
cd hafta-2
python -m pip install -r requirements.txt
python -m notebook backpropagation_calismasi.ipynb
python -m unittest -v
python train.py
```

## Çalışma günlüğü

Görevler ilerledikçe aşağıdaki tablo güncellenecek:

| Görev | Durum | Doğrulama | Not |
|---|---|---|---|
| 1. Value ve graph | Tamamlandı | Toplama, çarpma ve graph metadata test edildi | Ara node'lar korunuyor |
| 2. Elle backpropagation | Tamamlandı | Basit ifade ve tek nöron tabloları | Chain rule elle uygulandı |
| 3. Otomatik backward | Tamamlandı | Manuel değerlerle ve tekrar kullanılan node ile test | Gradient'ler `+=` ile birikiyor |
| 4. Üçlü gradient kontrolü | Tamamlandı | Micrograd, numerical ve PyTorch eşleşti | En büyük fark yaklaşık `4e-12` |
| 5. MLP eğitimi | Tamamlandı | 7 test ve loss eğrisi | Loss `5.23`ten `0.027`ye düştü |

## Doğrulanan sonuçlar

- Basit ifade gradient'leri: `a=6`, `b=-4`, `c=-2`, `f=4`
- Tek nöron output'u: `0.70710678`
- Atomic ve parçalanmış `tanh` input gradient'i: `0.5`
- Micrograd ve PyTorch gradient farkı: `0.0`
- Micrograd ve numerical derivative farkı: yaklaşık `4e-12`
- `3→4→4→1` MLP parametre sayısı: `41`
- Eğitim loss'u: `5.2305 → 0.0272`
- Otomatik testler: `7/7 başarılı`

## Kavram kontrol listesi

Bu çalışmanın sonunda aşağıdaki sorular cevaplanabilir:

- `Value` neden yalnızca sayıyı değil, graph bağlantılarını da saklıyor?
- Local derivative ile nihai loss gradient'i arasındaki fark ne?
- Topological sıra neden gerekli?
- Gradient'ler neden `=` ile yazılmıyor, `+=` ile biriktiriliyor?
- Numerical derivative neden eğitim için değil, doğrulama için kullanılıyor?
- `zero_grad` unutulursa ne olur?
- Backpropagation ile gradient descent arasındaki fark ne?
