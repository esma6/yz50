"""Scalar automatic differentiation motoru.

Bu modül Karpathy'nin micrograd fikrinin eğitim amaçlı küçük bir uygulamasıdır.
Her ``Value`` hem forward pass değerini hem de kendisini üreten computation
graph bağlantılarını saklar. ``backward()`` chain rule'u graph üzerinde ters
topological sırada uygulayarak bütün gradient'leri hesaplar.
"""

import math


class Value:
    """Bir scalar değeri ve bu değerin computation graph bilgisini saklar."""

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label

        # Her operasyon oluşturduğu output'un içine kendi local backward
        # fonksiyonunu yerleştirir. Leaf node'larda yapılacak işlem yoktur.
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"

    @staticmethod
    def _as_value(value):
        """Python int/float değerlerini gerektiğinde Value'ya dönüştür."""
        return value if isinstance(value, Value) else Value(value)

    def __add__(self, other):
        other = self._as_value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(self+other)/dself = 1 ve d(...)/dother = 1.
            # += önemlidir: aynı node'a farklı yollardan gradient gelebilir.
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = self._as_value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # d(self*other)/dself = other, tersi de self değeridir.
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        """Hyperbolic tangent aktivasyonu ve local derivative'i."""
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            # tanh'(x) = 1 - tanh(x)^2. Burada tanh(x), out.data=t.
            self.grad += (1.0 - t**2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        """e^x operasyonu; türevi yine e^x'tir."""
        result = math.exp(self.data)
        out = Value(result, (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, exponent):
        """Sabit bir kuvvete yükseltme: d(x^n)/dx = n*x^(n-1)."""
        if not isinstance(exponent, (int, float)):
            raise TypeError("Kuvvet sabit bir int veya float olmalıdır")
        out = Value(self.data**exponent, (self,), f"**{exponent}")

        def _backward():
            local_derivative = exponent * self.data ** (exponent - 1)
            self.grad += local_derivative * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-self._as_value(other))

    def __truediv__(self, other):
        return self * self._as_value(other) ** -1

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return self._as_value(other) - self

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return self._as_value(other) / self

    def backward(self):
        """Bu output'tan erişilen bütün node'ların gradient'lerini hesapla."""
        topological_order = []
        visited = set()

        def build_topological_order(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topological_order(child)
                topological_order.append(node)

        build_topological_order(self)

        # d(output)/d(output) = 1, backward pass'in başlangıç tohumu.
        self.grad = 1.0

        # Forward topological sırayı ters çevirince output'tan leaf'lere gideriz.
        for node in reversed(topological_order):
            node._backward()

    def zero_grad(self):
        """Tek bir Value'nun gradient'ini temizle."""
        self.grad = 0.0


def numerical_derivative(function, value, epsilon=1e-6):
    """Merkezi farkla f'(value) için sayısal yaklaşık değer hesapla."""
    return (
        function(value + epsilon) - function(value - epsilon)
    ) / (2.0 * epsilon)
