"""YZ50 first-week exercises implemented with only the Python standard library."""


def neuron_forward(inputs, weights, bias):
    """Return the weighted sum of inputs plus a bias term."""
    if len(inputs) != len(weights):
        raise ValueError("inputs and weights must have the same length")
    return sum(x * w for x, w in zip(inputs, weights)) + bias


def layer_forward(inputs, layer_weights, layer_biases):
    """Run the same inputs through every neuron in a layer."""
    if len(layer_weights) != len(layer_biases):
        raise ValueError("each neuron must have one bias")
    return [
        neuron_forward(inputs, weights, bias)
        for weights, bias in zip(layer_weights, layer_biases)
    ]


def mean_squared_error(predictions, targets):
    """Average squared distance between predictions and target values."""
    if not predictions or len(predictions) != len(targets):
        raise ValueError("predictions and targets must be non-empty and equally sized")
    squared_errors = [
        (prediction - target) ** 2
        for prediction, target in zip(predictions, targets)
    ]
    return sum(squared_errors) / len(squared_errors)


def dataset_predictions(xs, weight, bias):
    """Predict y = weight*x + bias for every scalar input."""
    return [neuron_forward([x], [weight], bias) for x in xs]


def dataset_loss(xs, targets, weight, bias):
    return mean_squared_error(dataset_predictions(xs, weight, bias), targets)


def numerical_derivative(function, value, epsilon=1e-5):
    """Estimate a derivative using the symmetric finite-difference formula."""
    return (function(value + epsilon) - function(value - epsilon)) / (2 * epsilon)


def gradient_descent(xs, targets, initial_weight, bias, learning_rate, steps):
    """Optimize one weight using only a numerical derivative."""
    weight = initial_weight
    history = []

    for step in range(steps + 1):
        loss = dataset_loss(xs, targets, weight, bias)
        history.append((step, weight, loss))
        if step == steps:
            break

        loss_for_weight = lambda candidate: dataset_loss(
            xs, targets, candidate, bias
        )
        gradient = numerical_derivative(loss_for_weight, weight)
        weight = weight - learning_rate * gradient

    return weight, history
