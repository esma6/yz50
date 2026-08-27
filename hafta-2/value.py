"""Hafta 2 — Value ve autograd motoru.

Bu dosya Görev 1 ile başlayarak adım adım doldurulacak.

İlk çalışma sırası:
1. Scalar ``data`` saklayan Value sınıfını yaz.
2. Toplama işlemini ekle.
3. Çarpma işlemini ekle.
4. Her yeni Value içinde onu üreten node'ları ve operasyonu sakla.

Kod yazmadan önce cevaplanacak soru:
Bir sonuç değerinden geriye doğru gidebilmek için forward pass sırasında hangi
bilgileri kaybetmemeliyim?
"""

class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        out = Value(
            self.data + other.data,
            (self, other),
            "+",
        )
        return out

    def __repr__(self):
        return f"Value(data={self.data})"