import unittest

from neural_network import (
    gradient_descent,
    layer_forward,
    mean_squared_error,
    neuron_forward,
    numerical_derivative,
)


class NeuralNetworkTests(unittest.TestCase):
    def test_neuron_forward(self):
        self.assertAlmostEqual(neuron_forward([1, 2], [3, 4], 5), 16)

    def test_layer_forward(self):
        self.assertEqual(layer_forward([1, 2], [[1, 0], [0, 1]], [0, 0]), [1, 2])

    def test_mean_squared_error(self):
        self.assertAlmostEqual(mean_squared_error([1, 3], [2, 1]), 2.5)

    def test_numerical_derivative(self):
        self.assertAlmostEqual(numerical_derivative(lambda x: x**2, 3), 6, places=6)

    def test_gradient_descent_reduces_loss(self):
        _, history = gradient_descent([-1, 0, 1], [-2, 0, 2], -1, 0, 0.1, 50)
        self.assertLess(history[-1][2], history[0][2])
        self.assertAlmostEqual(history[-1][1], 2, places=2)


if __name__ == "__main__":
    unittest.main()
