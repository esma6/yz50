"""Hafta 2 autograd ve neural-network doğrulama testleri."""

import math
import unittest

from nn import MLP
from train import train_model
from value import Value, numerical_derivative


class ValueTests(unittest.TestCase):
    def test_forward_graph_metadata(self):
        a, b = Value(2), Value(-3)
        result = a * b + Value(10)
        self.assertEqual(result.data, 4)
        self.assertEqual(result._op, "+")

    def test_simple_expression_manual_gradients(self):
        a, b, c, f = Value(2), Value(-3), Value(10), Value(-2)
        loss = (a * b + c) * f
        loss.backward()
        self.assertAlmostEqual(a.grad, 6)
        self.assertAlmostEqual(b.grad, -4)
        self.assertAlmostEqual(c.grad, -2)
        self.assertAlmostEqual(f.grad, 4)

    def test_reused_value_accumulates_gradient(self):
        a = Value(3)
        result = a * a
        result.backward()
        self.assertAlmostEqual(a.grad, 6)

    def test_tanh_atomic_and_decomposed_match(self):
        atomic_input = Value(0.8813735870195432)
        atomic_output = atomic_input.tanh()
        atomic_output.backward()

        decomposed_input = Value(0.8813735870195432)
        exp_2x = (2 * decomposed_input).exp()
        decomposed_output = (exp_2x - 1) / (exp_2x + 1)
        decomposed_output.backward()

        self.assertAlmostEqual(atomic_output.data, decomposed_output.data)
        self.assertAlmostEqual(atomic_input.grad, decomposed_input.grad)

    def test_autograd_matches_numerical_derivative(self):
        x = Value(0.5)
        output = (x * x + 2 * x).tanh()
        output.backward()
        numeric = numerical_derivative(
            lambda candidate: math.tanh(candidate**2 + 2 * candidate), 0.5
        )
        self.assertAlmostEqual(x.grad, numeric, places=6)


class NeuralNetworkTests(unittest.TestCase):
    def test_parameter_count(self):
        model = MLP(3, [4, 4, 1])
        self.assertEqual(len(model.parameters()), 41)

    def test_training_reduces_loss(self):
        _, history, _ = train_model(steps=50, verbose=False)
        self.assertLess(history[-1], history[0])
        self.assertLess(history[-1], 0.1)


if __name__ == "__main__":
    unittest.main()
