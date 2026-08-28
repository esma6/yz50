# YZ50 Çalışma Defteri

Bu repo, YZ50 boyunca her haftanın kodlarını, deney çıktılarını ve ders
notlarını aynı düzen içinde saklar. Her haftanın çalışması kendi klasöründedir;
böylece yeni haftalar eklendikçe önceki çalışmalar karmaşıklaşmaz.

## Haftalar

<details open>
<summary><strong>Hafta 1 — Neural Networks Temelleri</strong></summary>

- Tek nöron ve katman forward pass
- Mean Squared Error loss
- Parametre–loss ilişkisi ve loss eğrisi
- Sayısal türevle gradient descent
- Durum: **Tamamlandı**
- [Kodlar ve ayrıntılı ders notları](hafta-1/README.md)

</details>

<details open>
<summary><strong>Hafta 2 — Backpropagation ve Micrograd</strong></summary>

- `Value` sınıfı ve computation graph
- Local derivative ve chain rule
- Ters topological sırada `backward()`
- Numerical derivative ve PyTorch ile doğrulama
- `Neuron`, `Layer` ve `MLP` eğitimi
- Durum: **Tamamlandı — notebook çıktıları ve 7 doğrulama testi mevcut**
- [Çalışma planı ve başlangıç dosyaları](hafta-2/README.md)

</details>

## Standart klasör düzeni

Her yeni hafta aynı düzeni takip eder:

```text
hafta-N/
├── README.md       # Kavramlar, görevler, sonuçlar ve zorlanılan noktalar
├── kaynak kodlar   # O haftanın uygulaması
├── testler         # Matematiksel ve davranışsal doğrulamalar
└── outputs/        # Varsa grafik, CSV veya diğer deney çıktıları
```

Bir haftanın README dosyası, o haftayı tek başına anlayabilmek için yeterli
olacak şekilde tutulur. Kök README ise yalnızca genel yönlendirme yapar.

## Çalışma ilkesi

Her görevde aynı sıra izlenir:

```text
Kavramı öğren → Küçük örneği elle çöz → Kodu yaz → Sayısal olarak doğrula
→ Test ekle → Sonucu ve zorlanılan noktayı README'ye kaydet
```

Bu sayede repo yalnızca tamamlanmış ödevlerin değil, öğrenme sürecinin de kaydı
olur.
