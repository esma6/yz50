"""Hafta 2 çalışma notebook'unu temiz UTF-8 içerikle yeniden üretir."""

from pathlib import Path

import nbformat as nbf


NOTEBOOK_PATH = Path(__file__).with_name("backpropagation_calismasi.ipynb")


def markdown(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    markdown(r"""
# YZ50 — Hafta 2: Backpropagation Çalışma Defteri

Bir sinir ağı bir tahmin ürettiğinde yalnızca ne kadar hata yaptığını bilmek
yeterli değildir. Asıl ihtiyaç, bu hatada her ağırlığın ne kadar payı olduğunu
bulmaktır. Backpropagation bu soruyu, sonucu oluşturan işlemleri geriye doğru
izleyerek cevaplar.

Bu defter Karpathy'nin micrograd yaklaşımını scalar seviyede yeniden kuruyor.
Amaç yalnızca çalışan bir `backward()` metodu elde etmek değil; computation
graph, local derivative, chain rule ve gradient descent arasındaki bağlantıyı
adım adım görünür hale getirmek. Kod hücrelerinin çıktıları kaydedildiği için
hesapları yeniden çalıştırmadan da düşünce akışı takip edilebilir.
"""),
    markdown(r"""
## Büyük resim

Geçen hafta bir parametrenin türevini sayısal olarak, fonksiyonu birkaç kez
çalıştırarak hesapladık. Bu hafta bütün parametrelerin gradient'lerini tek bir
geriye geçişte hesaplayacağız:

```text
Forward pass
    ↓
Computation graph
    ↓
Final loss
    ↓
Backward pass: local derivative × üstten gelen gradient
    ↓
Bütün weight ve bias gradient'leri
    ↓
Gradient descent update
```

Önemli ayrım:

- **Backpropagation**, gradient'leri hesaplar.
- **Gradient descent**, hesaplanan gradient'lerle parametreleri günceller.
"""),
    code(r"""
%load_ext autoreload
%autoreload 2

import math
import random
import sys

import matplotlib.pyplot as plt

from value import Value, numerical_derivative
from nn import Neuron, Layer, MLP
from train import TRAINING_INPUTS, TARGETS, train_model

print("Python:", sys.version.split()[0])
"""),
    markdown(r"""
# 1 — Bir sayı geçmişini hatırlarsa: `Value` ve computation graph

## 1.1 Neden normal sayı yerine `Value`?

Normal bir `float` yalnızca sayısal sonucu saklar. Backpropagation yapabilmek
için bir sonucun:

- Sayısal değerini (`data`)
- Kendisini üreten node'ları (`_prev`)
- Hangi işlemle üretildiğini (`_op`)
- Final output'a göre türevini (`grad`)

saklamamız gerekir.

Doğrudan oluşturulan ve başka bir işlemden gelmeyen node'a **leaf node** denir.
Leaf node'un `_prev` kümesi ve `_op` bilgisi boştur.
"""),
    code(r"""
a = Value(2.0, label="a")
b = Value(-3.0, label="b")

print("a:", a)
print("b:", b)
print("a leaf mi?:", len(a._prev) == 0 and a._op == "")
"""),
    markdown(r"""
### Henüz yalnızca bir sayı var

Ekrandaki `data` alanı forward pass sırasında kullanılan sayıdır. `grad` ise
henüz sıfırdır; çünkü hangi sonucu inceleyeceğimize karar vermedik ve geriye
doğru bir hesap başlatmadık. `a` ile `b` başka işlemlerden doğmadığı için
graph'ın başlangıç noktaları, yani leaf node'larıdır.

Tek başına bu yapı normal bir sayıdan çok farklı görünmüyor. Fark, iki `Value`
arasında işlem yaptığımız anda ortaya çıkacak: yeni sonuç, nereden geldiğini de
hatırlayacak.
"""),
    markdown(r"""
## 1.2 Toplama ile ilk graph bağlantısı

Elle hesap:

```text
c = a + b = 2 + (-3) = -1
```

Beklenen graph:

```text
a ──┐
    + ──> c
b ──┘
```

Yeni `c`, yalnızca `-1` değerini değil, `a` ve `b` üzerinden `+` işlemiyle
üretildiğini de saklamalıdır.
"""),
    code(r"""
c = a + b
c.label = "c"

print("c:", c)
print("operasyon:", c._op)
print("a önceki node mu?:", a in c._prev)
print("b önceki node mu?:", b in c._prev)
"""),
    markdown(r"""
### Sonuç artık nereden geldiğini biliyor

`c.data=-1` bildiğimiz toplama sonucudur. Yeni olan, `_op='+'` bilgisi ve
`_prev` içinde duran iki bağlantıdır. Böylece `c` yalnızca `-1` sayısı değildir;
“`a` ile `b` toplandığında oluşan `-1`”dir. Backward pass daha sonra tam olarak
bu bağlantıları ters yönde izleyecek.
"""),
    markdown(r"""
## 1.3 Çarpma ve birleşik ifade

Elle hesap:

```text
e = a × b = 2 × (-3) = -6
d = e + 10 = -6 + 10 = 4
```

```text
a ──┐
    * ──> e ──┐
b ──┘         + ──> d
10 ───────────┘
```
"""),
    code(r"""
ten = Value(10.0, label="ten")
e = a * b
e.label = "e"
d = e + ten
d.label = "d"

for node in [a, b, e, ten, d]:
    previous_labels = sorted(child.label for child in node._prev)
    print(
        f"{node.label:>3} | data={node.data:>5.1f} | "
        f"op={node._op or 'leaf':>4} | prev={previous_labels}"
    )
"""),
    markdown(r"""
### Forward pass aslında iki şey üretir

Toplama ve çarpma yeni bir `Value` oluştururken ara sonuçları kaybetmez.
Final `d` node'undan `e` ve `ten`e, `e` üzerinden de `a` ve `b`ye geri
ulaşabiliriz. Bu bağlantılar daha sonra backward pass'in izleyeceği yolları
oluşturur.

Forward pass'in iki çıktısı vardır:

1. Hesabın sayısal sonucu
2. Bu hesabın computation graph'ı

Graph'ı kurmak tek başına öğrenme sağlamaz. Şimdi bu bağlantılara ikinci bir
anlam yüklememiz gerekiyor: final sonuçtaki küçük bir değişimin her ara değerle
nasıl ilişkili olduğunu, yani gradient'leri bulacağız.
"""),
    markdown(r"""
# 2 — Graph üzerinde geriye yürümek: gradient ve chain rule

## 2A — Basit ifade

Kullanacağımız ifade:

```text
e = a × b
d = e + c
L = d × f
```

Değerler ve forward pass:

```text
a=2, b=-3, c=10, f=-2
e=-6, d=4, L=-8
```

Her node'un `grad` alanında şu türevi saklayacağız:

```text
node.grad = ∂L/∂node
```
"""),
    code(r"""
a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(10.0, label="c")
f = Value(-2.0, label="f")
e = a * b; e.label = "e"
d = e + c; d.label = "d"
L = d * f; L.label = "L"

print(f"e={e.data}, d={d.data}, L={L.data}")
"""),
    markdown(r"""
### Geriye doğru hesabın başlangıç noktası

Her zaman final output'tan başlarız:

```text
∂L/∂L = 1
```

Bu, “L'yi çok küçük miktarda değiştirirsem L aynı miktarda değişir” demektir.
"""),
    code(r"""
L.grad = 1.0
"""),
    markdown(r"""
### İlk kapı: `L = d × f`

Çarpmanın local derivative'leri:

```text
∂L/∂d = f = -2
∂L/∂f = d = 4
```

Genel backward kuralında local derivative, üstten gelen `L.grad` ile çarpılır.
Bu ilk node'da `L.grad=1`dir.
"""),
    code(r"""
d.grad = f.data * L.grad
f.grad = d.data * L.grad
"""),
    markdown(r"""
### Gradient'in iki kola ayrılması: `d = e + c`

Toplamanın iki local derivative'i de `1`dir. Chain rule:

```text
∂L/∂e = ∂L/∂d × ∂d/∂e = -2 × 1 = -2
∂L/∂c = ∂L/∂d × ∂d/∂c = -2 × 1 = -2
```

Toplama node'u üstten gelen gradient'i iki kola aynen yönlendirir.
"""),
    code(r"""
e.grad = 1.0 * d.grad
c.grad = 1.0 * d.grad
"""),
    markdown(r"""
### Leaf node'lara ulaşmak: `e = a × b`

```text
∂e/∂a = b = -3
∂e/∂b = a = 2
```

Chain rule:

```text
∂L/∂a = ∂L/∂e × ∂e/∂a = -2 × -3 = 6
∂L/∂b = ∂L/∂e × ∂e/∂b = -2 × 2 = -4
```
"""),
    code(r"""
a.grad = b.data * e.grad
b.grad = a.data * e.grad

for node in [a, b, c, e, d, f, L]:
    print(f"{node.label:>2} | data={node.data:>5.1f} | grad={node.grad:>5.1f}")
"""),
    markdown(r"""
## Gradient sayı olmaktan çıkarılıp anlamlandırıldığında

- `a.grad=6`: `a`yı küçük bir `h` kadar artırırsak `L` yaklaşık `6h` artar.
- `b.grad=-4`: `b`yi artırırsak `L` yaklaşık dört kat hızlı azalır.
- `c.grad=-2`: `c`yi artırmak `L`yi azaltır.
- `f.grad=4`: `f`yi artırmak `L`yi artırır.

Gradient'in **işareti**, etkinin yönünü; **mutlak büyüklüğü**, hassasiyetin
gücünü gösterir. Bunlar yalnızca mevcut değerlerin yakınındaki yerel bilgilerdir.

Backpropagation'ın özeti:

```text
child.grad += local_derivative × output.grad
```
"""),
    markdown(r"""
## 2B — Tek nöronda manuel backpropagation

İki girdili nöron:

```text
n = x1*w1 + x2*w2 + b
o = tanh(n)
```

Değerler:

```text
x1=2, x2=0, w1=-3, w2=1, b≈6.88137
n≈0.88137, o≈0.7071
```

`tanh` türevi:

```text
d tanh(n)/dn = 1 - tanh(n)^2 = 1 - o^2
```
"""),
    code(r"""
x1 = Value(2.0, label="x1")
x2 = Value(0.0, label="x2")
w1 = Value(-3.0, label="w1")
w2 = Value(1.0, label="w2")
bias = Value(6.8813735870195432, label="b")

x1w1 = x1 * w1; x1w1.label = "x1w1"
x2w2 = x2 * w2; x2w2.label = "x2w2"
n = x1w1 + x2w2 + bias; n.label = "n"
o = n.tanh(); o.label = "o"

print(f"n={n.data:.8f}")
print(f"o=tanh(n)={o.data:.8f}")
"""),
    markdown(r"""
### Nöron gradient'lerini elle doldurmak

1. `o.grad = 1`
2. `n.grad = (1-o²) × o.grad = 0.5`
3. Toplamalar `0.5` gradient'ini `x1w1`, `x2w2` ve `bias`a taşır.
4. Çarpma kollarında diğer girdinin değeri local derivative olur.

Beklenen leaf gradient'leri:

```text
x1.grad = 0.5*w1 = -1.5
w1.grad = 0.5*x1 = 1.0
x2.grad = 0.5*w2 = 0.5
w2.grad = 0.5*x2 = 0.0
b.grad  = 0.5
```

`w2.grad=0` özellikle anlamlıdır: `x2=0` olduğu için bu örnekte `w2` ne kadar
değişirse değişsin `x2*w2` sıfır kalır ve output'u etkilemez.
"""),
    code(r"""
o.grad = 1.0
n.grad = (1.0 - o.data**2) * o.grad
x1w1.grad = n.grad
x2w2.grad = n.grad
bias.grad = n.grad
x1.grad = w1.data * x1w1.grad
w1.grad = x1.data * x1w1.grad
x2.grad = w2.data * x2w2.grad
w2.grad = x2.data * x2w2.grad

for node in [x1, w1, x2, w2, bias, n, o]:
    print(f"{node.label:>2} | data={node.data:>9.6f} | grad={node.grad:>9.6f}")
"""),
    markdown(r"""
### Bir nöron da aynı graph fikrinden oluşur

Nöron ilk bakışta yeni ve daha karmaşık bir yapı gibi görünse de aslında
toplama, çarpma ve `tanh` node'larından oluşan başka bir matematiksel ifadedir.
Her node yalnızca kendi local derivative'ini bilir; chain rule bu küçük yerel
bilgileri final output'a kadar bağlar.

Gradient'leri elle doldurmak mekanizmayı görünür yaptı, fakat büyük bir graph'ta
her node'u elle gezmek mümkün değildir. Bir sonraki bölümde aynı hesabın sırasını
bozmadan otomatikleştireceğiz.
"""),
    markdown(r"""
# 3 — Elle yaptığımız yürüyüşü otomatikleştirmek: `backward()`

Elle yaptığımız işlemi otomatikleştirmek için:

1. Output'tan erişilebilen bütün node'ları buluruz.
2. Çocuklar kendilerini kullanan node'lardan önce gelecek şekilde topological
   sıra oluştururuz.
3. Output gradient'ini `1` yaparız.
4. Sırayı ters çevirip her node'un `_backward()` fonksiyonunu çalıştırırız.

Ters sıra gereklidir çünkü bir node backward olmadan önce yukarıdaki bütün
yollardan gelen gradient katkılarının onda toplanmış olması gerekir.
"""),
    code(r"""
a2 = Value(2.0, label="a")
b2 = Value(-3.0, label="b")
c2 = Value(10.0, label="c")
f2 = Value(-2.0, label="f")
L2 = (a2 * b2 + c2) * f2
L2.label = "L"

L2.backward()

print("Otomatik backward sonuçları:")
for node in [a2, b2, c2, f2, L2]:
    print(f"{node.label:>2} | data={node.data:>5.1f} | grad={node.grad:>5.1f}")
"""),
    markdown(r"""
### Otomasyonun doğru olduğuna dair ilk kanıt

| Node | Manuel gradient | Otomatik gradient |
|---|---:|---:|
| a | 6 | 6 |
| b | -4 | -4 |
| c | -2 | -2 |
| f | 4 | 4 |

Eşleşme, topological traversal ve local backward kurallarının elle yaptığımız
chain rule hesabını doğru şekilde otomatikleştirdiğini gösterir.
"""),
    markdown(r"""
## Gradient neden `+=` ile biriktirilir?

Bir node graph içinde birden fazla yerde kullanılabilir:

```text
y = x × x
```

`x`, output'u iki ayrı yol üzerinden etkiler. `x=3` için:

```text
dy/dx = x + x = 6
```

İki yolun katkısını toplamaz, gradient'in üzerine yazarsak yanlışlıkla `3`
buluruz. Bu nedenle local backward kurallarında `=` değil `+=` kullanılır.
"""),
    code(r"""
reused = Value(3.0, label="x")
squared = reused * reused
squared.backward()

print("x=3 için x*x gradient'i:", reused.grad)
print("Beklenen analitik değer 2*x:", 2 * reused.data)
"""),
    markdown(r"""
### `backward()` yalnızca sinir ağlarına ait değildir

`backward()` özel bir neural-network işlemi değildir. Herhangi bir scalar
computation graph üzerinde chain rule'u ters sırada uygular. Neural network'ler
bu genel matematiksel ifadelerin özel ve büyük bir sınıfıdır.

Elle bulunan değerlerle eşleşmek iyi bir başlangıçtır; yine de uygulamanın
kendi varsayımlarını kullanarak kendisini doğrulaması yeterli değildir. Şimdi
aynı türevi farklı graph yapıları ve bağımsız yöntemlerle karşılaştıracağız.
"""),
    markdown(r"""
# 4 — Aynı matematiğe farklı yollardan ulaşmak: `tanh` ve gradient kontrolü

Hyperbolic tangent daha küçük operasyonlarla yazılabilir:

```text
tanh(x) = (e^(2x)-1) / (e^(2x)+1)
```

Bunun için `exp`, sabit `pow`, negatif alma, çıkarma ve bölme operasyonlarını
ekledik. Bir operasyonu tek node veya küçük node'ların birleşimi olarak yazmak
forward sonucu ve leaf gradient'lerini değiştirmemelidir.
"""),
    code(r"""
input_value = 0.8813735870195432

atomic_x = Value(input_value, label="atomic_x")
atomic_output = atomic_x.tanh()
atomic_output.backward()

decomposed_x = Value(input_value, label="decomposed_x")
exp_2x = (2 * decomposed_x).exp()
decomposed_output = (exp_2x - 1) / (exp_2x + 1)
decomposed_output.backward()

print(f"Atomic tanh output:      {atomic_output.data:.12f}")
print(f"Parçalanmış output:      {decomposed_output.data:.12f}")
print(f"Atomic input gradient:   {atomic_x.grad:.12f}")
print(f"Parçalanmış gradient:    {decomposed_x.grad:.12f}")
"""),
    markdown(r"""
### Graph değişse de temsil edilen fonksiyon değişmedi

İki graph'ın iç node'ları farklıdır fakat temsil ettikleri matematiksel
fonksiyon aynıdır. Autograd açısından bir operasyonun ne kadar “atomik” olduğu
tasarım tercihidir. Gerekli olan yalnızca doğru forward hesap ve doğru local
derivative'tir.

## Üçlü gradient kontrolü

Aynı fonksiyonu üç yöntemle hesaplayacağız:

```text
f(x) = tanh(x² + 2x), x=0.5
```

1. Kendi `Value.backward()` motorumuz
2. Geçen haftaki merkezi fark numerical derivative
3. PyTorch autograd

Sayısal türev eğitim için verimsizdir; burada analitik backward uygulamamızı
kontrol eden bağımsız bir referans olarak kullanılır.
"""),
    code(r"""
x_value = 0.5

# 1) Kendi autograd motorumuz
micro_x = Value(x_value)
micro_result = (micro_x * micro_x + 2 * micro_x).tanh()
micro_result.backward()

# 2) Numerical derivative
plain_function = lambda x: math.tanh(x*x + 2*x)
numeric_gradient = numerical_derivative(plain_function, x_value)

# 3) PyTorch autograd
try:
    import torch
    torch_x = torch.tensor(x_value, dtype=torch.float64, requires_grad=True)
    torch_result = torch.tanh(torch_x * torch_x + 2 * torch_x)
    torch_result.backward()
    torch_gradient = torch_x.grad.item()
except ImportError:
    torch_gradient = float("nan")
    print("PyTorch kurulu değil; üçüncü karşılaştırma çalıştırılamadı.")

print(f"Micrograd gradient: {micro_x.grad:.12f}")
print(f"Numerical gradient: {numeric_gradient:.12f}")
print(f"PyTorch gradient:   {torch_gradient:.12f}")
print(f"Micro-numeric fark: {abs(micro_x.grad-numeric_gradient):.3e}")
print(f"Micro-torch fark:   {abs(micro_x.grad-torch_gradient):.3e}")
"""),
    markdown(r"""
### Üç yöntemin aynı sayıda buluşması neyi gösteriyor?

Üç sonucun floating-point toleransı içinde eşleşmesi şunları doğrular:

- Local derivative formüllerimiz doğru.
- Chain rule doğru yönde uygulanıyor.
- Topological backward sırası doğru.
- Birden fazla gradient yolu doğru toplanıyor.

Küçük son basamak farkları hata değil, floating-point temsilinin sonucudur.
Bu karşılaştırmadan sonra `Value` motorunun yalnızca birkaç seçilmiş örnekte
çalıştığını değil, türev kurallarını tutarlı biçimde uyguladığını söyleyebiliriz.
Artık bu motoru tek tek ifadeler yerine katmanlı bir modelin altında kullanmaya
hazırız.
"""),
    markdown(r"""
# 5 — Küçük işlemlerden öğrenen bir modele: `Neuron`, `Layer` ve `MLP`

Yapı:

```text
Value → Neuron → Layer → MLP
```

Bir nöron:

```text
output = tanh(w₁x₁ + w₂x₂ + ... + b)
```

Bir layer aynı input'ları bağımsız parametrelere sahip birden fazla nörona
verir. MLP ise layer'ları sırayla birbirine bağlar.
"""),
    code(r"""
random.seed(42)

neuron = Neuron(3)
layer = Layer(3, 4)
model = MLP(3, [4, 4, 1])
sample = [2.0, 3.0, -1.0]

print("Tek nöron output'u:", neuron(sample))
print("Dört nöronlu layer output'ları:", layer(sample))
print("MLP output'u:", model(sample))
print("3→4→4→1 MLP parametre sayısı:", len(model.parameters()))
"""),
    markdown(r"""
## Neden 41 parametre var?

```text
3→4 layer: 4 × (3 weight + 1 bias) = 16
4→4 layer: 4 × (4 weight + 1 bias) = 20
4→1 layer: 1 × (4 weight + 1 bias) = 5
Toplam: 16 + 20 + 5 = 41
```

Input verileri parametre değildir. Eğitim sırasında değiştirilen değerler
weight ve bias'lardır.
"""),
    markdown(r"""
## Küçük veri kümesi ve loss

Hedefler `[-1, 1]` aralığındadır; output'ta `tanh` kullandığımız için bu aralık
uygundur. Loss, dört örneğin squared error toplamıdır:

```text
loss = Σ(prediction-target)²
```

Loss küçüldükçe tahminler hedeflere yaklaşır.
"""),
    code(r"""
for inputs, target in zip(TRAINING_INPUTS, TARGETS):
    print("input:", inputs, "target:", target)
"""),
    markdown(r"""
## Doğru training loop sırası

```text
1. Forward pass
2. Loss hesapla
3. zero_grad
4. backward
5. parameter.data -= learning_rate × parameter.grad
```

`zero_grad`, her yeni training step'inde eski gradient'leri temizler.
Backward kuralları aynı graph içindeki yolları toplayabilmek için `+=` kullanır;
bu nedenle farklı training step'leri arasında temizleme yapmazsak eski
gradient'ler yanlışlıkla birikir.
"""),
    code(r"""
# zero_grad unutulduğunda gradient birikimini küçük örnekte görelim.
p = Value(2.0)
first_loss = p * p
first_loss.backward()
first_gradient = p.grad

second_loss = p * p
second_loss.backward()
accumulated_gradient = p.grad

p.grad = 0.0
third_loss = p * p
third_loss.backward()
reset_gradient = p.grad

print("İlk backward gradient'i:       ", first_gradient)
print("Sıfırlamadan ikinci backward:  ", accumulated_gradient)
print("Sıfırladıktan sonra backward:  ", reset_gradient)
"""),
    markdown(r"""
### Meşhur `zero_grad` bug'ı

`p²` için `p=2` noktasındaki doğru gradient `2p=4`tür. İkinci backward öncesi
gradient temizlenmezse önceki `4` ile yeni `4` toplanıp `8` olur. Basit bir
problem bu bug'a rağmen loss düşürebilir; “model çalışıyor” görünmesi kodun
doğru olduğuna kanıt değildir.
"""),
    code(r"""
trained_model, loss_history, final_predictions = train_model(
    steps=50,
    learning_rate=0.05,
    seed=42,
    verbose=True,
)

print("\nİlk loss:", loss_history[0])
print("Son loss: ", loss_history[-1])
print("Hedefler: ", TARGETS)
print("Tahminler:", [round(value, 4) for value in final_predictions])
"""),
    code(r"""
plt.figure(figsize=(8, 4.5))
plt.plot(loss_history, color="#2563eb", linewidth=2)
plt.title("MLP eğitimi sırasında loss")
plt.xlabel("Training adımı")
plt.ylabel("Squared error loss")
plt.grid(alpha=0.25)
plt.show()
"""),
    markdown(r"""
### Loss eğrisi modelin öğrendiğini nasıl gösteriyor?

- İlk loss, rastgele parametrelerin tahmin hatasıdır.
- Her backward pass 41 parametrenin gradient'ini hesaplar.
- Gradient descent bütün parametreleri loss'un azalış yönünde günceller.
- Loss eğrisinin düşmesi optimizasyonun çalıştığını gösterir.
- Son tahminlerin `[1, -1, -1, 1]` hedeflerine yaklaşması loss düşüşünün model
  davranışına yansıdığını gösterir.

Learning rate yalnızca yerel gradient bilgisine dayanarak ne kadar büyük adım
atılacağını belirler. Çok küçük değer yavaş, çok büyük değer kararsız eğitim
oluşturabilir.
"""),
    markdown(r"""
# Parçaları yeniden birleştirmek

Bu çalışmada:

1. Matematiksel işlemler yapılırken computation graph kurdum.
2. Basit ifade ve tek nöronda gradient'leri elle hesapladım.
3. Chain rule'u ters topological sırada uygulayan `backward()` yazdım.
4. Aynı node'un birden fazla kullanıldığı durumda gradient'leri topladım.
5. `tanh`ı `exp`, `pow`, çıkarma ve bölme işlemlerine ayırdım.
6. Gradient'leri kendi backward motorum, numerical derivative ve PyTorch ile
   karşılaştırdım.
7. `Neuron`, `Layer` ve `MLP` sınıflarını kurup 41 parametreli ağı eğittim.
8. Her training adımında gradient'lerin neden sıfırlanması gerektiğini gördüm.

## Tek cümlelik zihinsel model

> Backpropagation, final loss'tan başlayarak her node'un local derivative'ini
> üstten gelen gradient ile çarpar, birden fazla yolun katkılarını toplar ve
> bütün parametrelerin loss üzerindeki etkisini tek bir geriye geçişte hesaplar.

## Language model bağlantısı

Büyük language model'ler farklı mimari, tensor işlemleri ve cross-entropy loss
kullanır. Ancak temel eğitim döngüsü aynıdır:

```text
token'lar → forward → next-token loss → backward → optimizer update
```

Micrograd scalar seviyede aynı mekanizmanın görünür ve anlaşılabilir halidir.
"""),
]


notebook = nbf.v4.new_notebook()
notebook["cells"] = cells
notebook["metadata"]["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
notebook["metadata"]["language_info"] = {
    "name": "python",
    "version": "3.12",
}

nbf.write(notebook, NOTEBOOK_PATH)
print(f"Notebook oluşturuldu: {NOTEBOOK_PATH}")
print(f"Hücre sayısı: {len(cells)}")
