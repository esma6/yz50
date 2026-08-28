"""Küçük bir veri kümesi üzerinde MLP eğitimi."""

import random

from nn import MLP


TRAINING_INPUTS = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
TARGETS = [1.0, -1.0, -1.0, 1.0]


def train_model(steps=50, learning_rate=0.05, seed=42, verbose=True):
    """3→4→4→1 MLP'yi eğit; model, loss geçmişi ve tahminleri döndür."""
    random.seed(seed)
    model = MLP(3, [4, 4, 1])
    history = []

    for step in range(steps):
        # 1) Forward pass: bütün örnekler için tahmin üret.
        predictions = [model(inputs) for inputs in TRAINING_INPUTS]

        # 2) Loss: squared error'ları topla. Düşük loss daha iyi tahmindir.
        loss = sum(
            (prediction - target) ** 2
            for prediction, target in zip(predictions, TARGETS)
        )
        history.append(loss.data)

        # 3) Çok önemli: önceki training step'in gradient'lerini temizle.
        model.zero_grad()

        # 4) Backward pass: bütün parametre gradient'lerini hesapla.
        loss.backward()

        # 5) Gradient descent: loss'un artış yönünün tersine git.
        for parameter in model.parameters():
            parameter.data -= learning_rate * parameter.grad

        if verbose and (step < 5 or (step + 1) % 10 == 0):
            print(f"adım={step:02d} loss={loss.data:.8f}")

    final_predictions = [model(inputs).data for inputs in TRAINING_INPUTS]
    return model, history, final_predictions


if __name__ == "__main__":
    trained_model, losses, outputs = train_model()
    print("\nParametre sayısı:", len(trained_model.parameters()))
    print("Hedefler:       ", TARGETS)
    print("Son tahminler:  ", [round(value, 4) for value in outputs])
