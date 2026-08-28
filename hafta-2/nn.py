"""Value autograd motoru üzerine kurulan küçük neural-network kütüphanesi."""

import random

from value import Value


class Neuron:
    """Ağırlıklı toplam, bias ve tanh aktivasyonundan oluşan tek nöron."""

    def __init__(self, number_of_inputs):
        self.weights = [
            Value(random.uniform(-1, 1), label=f"w{i}")
            for i in range(number_of_inputs)
        ]
        self.bias = Value(random.uniform(-1, 1), label="b")

    def __call__(self, inputs):
        weighted_sum = sum(
            (weight * input_value for weight, input_value in zip(self.weights, inputs)),
            self.bias,
        )
        return weighted_sum.tanh()

    def parameters(self):
        return self.weights + [self.bias]


class Layer:
    """Aynı input'ları bağımsız parametrelerle işleyen nöron grubu."""

    def __init__(self, number_of_inputs, number_of_outputs):
        self.neurons = [
            Neuron(number_of_inputs) for _ in range(number_of_outputs)
        ]

    def __call__(self, inputs):
        outputs = [neuron(inputs) for neuron in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self):
        return [
            parameter
            for neuron in self.neurons
            for parameter in neuron.parameters()
        ]


class MLP:
    """Katmanların sırayla birbirine bağlandığı multi-layer perceptron."""

    def __init__(self, number_of_inputs, layer_sizes):
        sizes = [number_of_inputs] + layer_sizes
        self.layers = [
            Layer(sizes[index], sizes[index + 1])
            for index in range(len(layer_sizes))
        ]

    def __call__(self, inputs):
        output = inputs
        for layer in self.layers:
            output = layer(output)
        return output

    def parameters(self):
        return [
            parameter
            for layer in self.layers
            for parameter in layer.parameters()
        ]

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.grad = 0.0
