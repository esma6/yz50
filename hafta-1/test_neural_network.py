import unittest

from neural_network import (
    gradient_descent,
    layer_forward,
    mean_squared_error,
    neuron_forward,
    numerical_derivative,
)


class NeuralNetworkTests(unittest.TestCase):
    """Ders notlarındaki sayısal iddiaların kodda da geçerli olduğunu doğrula."""

    def test_neuron_forward(self):
        # 1*3 + 2*4 + 5 bias = 16
        self.assertAlmostEqual(neuron_forward([1, 2], [3, 4], 5), 16)

    def test_layer_forward(self):
        # İlk nöron x1'i, ikinci nöron x2'yi aynen seçiyor.
        self.assertEqual(layer_forward([1, 2], [[1, 0], [0, 1]], [0, 0]), [1, 2])

    def test_mean_squared_error(self):
        # [(1-2)^2 + (3-1)^2] / 2 = (1+4)/2 = 2.5
        self.assertAlmostEqual(mean_squared_error([1, 3], [2, 1]), 2.5)

    def test_numerical_derivative(self):
        # f(x)=x^2 fonksiyonunun analitik türevi 2x'tir; x=3 için 6 beklenir.
        self.assertAlmostEqual(numerical_derivative(lambda x: x**2, 3), 6, places=6)

    def test_gradient_descent_reduces_loss(self):
        # Veri y=2x kuralına sahip. Eğitim loss'u düşürmeli ve weight'i 2'ye
        # yaklaştırmalıdır; böylece "öğrenme" iki ayrı iddiayla test edilir.
        _, history = gradient_descent([-1, 0, 1], [-2, 0, 2], -1, 0, 0.1, 50)
        self.assertLess(history[-1][2], history[0][2])
        self.assertAlmostEqual(history[-1][1], 2, places=2)


if __name__ == "__main__":
    unittest.main()
