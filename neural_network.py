"""Sinir ağlarının temel matematiğini gösteren küçük fonksiyonlar.

Bu dosyada hazır bir neural-network kütüphanesi özellikle kullanılmıyor. Amaç,
bir nöronun yaptığı ağırlıklı toplamı, loss hesabını ve gradient descent
güncellemesini Python işlemleri seviyesinde görebilmek.
"""


def neuron_forward(inputs, weights, bias):
    """Tek bir doğrusal nöronun forward pass sonucunu hesapla.

    Matematiksel karşılığı:
        output = x1*w1 + x2*w2 + ... + xn*wn + bias

    ``inputs`` veriyi, ``weights`` ve ``bias`` ise modelin parametrelerini
    temsil eder. Forward pass yalnızca mevcut parametrelerle çıktı üretir;
    bu aşamada öğrenme ya da parametre güncellemesi gerçekleşmez.

    Bu ilk hafta çalışmasında aktivasyon fonksiyonu kullanılmıyor. Dolayısıyla
    dönen değer, aktivasyon öncesi ham ağırlıklı toplamdır (çoğu kaynakta z).
    """
    if len(inputs) != len(weights):
        raise ValueError("Girdi ve ağırlık sayıları eşit olmalıdır")

    # zip, her girdiyi ona karşılık gelen weight ile eşleştirir.
    # Her x*w çarpımı o girdinin nöron çıktısına yaptığı katkıdır.
    weighted_sum = sum(x * w for x, w in zip(inputs, weights))

    # Bias girdilerden bağımsız bir kaydırmadır. Bütün girdiler sıfır olsa bile
    # nöronun sıfırdan farklı çıktı verebilmesini sağlar.
    return weighted_sum + bias


def layer_forward(inputs, layer_weights, layer_biases):
    """Aynı girdileri katmandaki bütün nöronlardan geçir.

    Bir layer, bağımsız nöronların bir araya gelmesidir. Her nöron aynı input
    listesini görür; fakat kendi weight ve bias değerlerine sahip olduğu için
    farklı bir çıktı üretir. Sonuç bu nedenle tek sayı değil, çıktı listesidir.
    """
    if len(layer_weights) != len(layer_biases):
        raise ValueError("Katmandaki her nöronun bir bias değeri olmalıdır")

    # layer_weights içindeki her alt liste tek bir nöronun weight'leridir.
    # Örneğin üç alt liste varsa katmanda üç nöron ve üç çıktı vardır.
    return [
        neuron_forward(inputs, weights, bias)
        for weights, bias in zip(layer_weights, layer_biases)
    ]


def mean_squared_error(predictions, targets):
    """Tahminlerle hedefler arasındaki Mean Squared Error loss'unu hesapla.

    Formül:
        MSE = sum((prediction - target)^2) / örnek_sayısı

    Farkların karesini almak pozitif ve negatif hataların birbirini götürmesini
    engeller. Loss sıfıra yaklaştıkça tahminler hedeflere yaklaşır.
    """
    if not predictions or len(predictions) != len(targets):
        raise ValueError("Tahminler ve hedefler boş olmamalı ve eşit sayıda olmalıdır")

    # Her veri örneğinin hatasını ayrı ayrı hesaplıyoruz.
    squared_errors = [
        (prediction - target) ** 2
        for prediction, target in zip(predictions, targets)
    ]

    # mean (ortalama), toplam kare hatayı örnek sayısına bölmek demektir.
    return sum(squared_errors) / len(squared_errors)


def dataset_predictions(xs, weight, bias):
    """Tek nöronu veri kümesindeki her x için çalıştır.

    Bu deneyde her örneğin tek girdisi olduğu için model ``y = weight*x +
    bias`` biçimindedir. Aynı öğrenilebilir weight ve bias bütün örneklerde
    tekrar kullanılır.
    """
    return [neuron_forward([x], [weight], bias) for x in xs]


def dataset_loss(xs, targets, weight, bias):
    """Belirli parametrelerin bütün veri kümesindeki loss'unu döndür."""
    # Önce forward pass ile tahminler, sonra hedeflerle MSE hesaplanır.
    # Eğitim döngüsünde azaltmaya çalıştığımız tek sayı budur.
    return mean_squared_error(dataset_predictions(xs, weight, bias), targets)


def numerical_derivative(function, value, epsilon=1e-5):
    """Merkezi fark yöntemiyle türevi sayısal olarak yaklaşık hesapla.

    Formül:
        f'(x) ≈ [f(x + epsilon) - f(x - epsilon)] / (2*epsilon)

    Bu değer, x çok küçük bir miktar değiştiğinde f'nin hangi yönde ve ne
    hızda değiştiğini söyler. Gerçek ağlarda her parametre için bu şekilde
    tekrar hesaplamak pahalı olduğundan backpropagation kullanılır.
    """
    loss_on_right = function(value + epsilon)
    loss_on_left = function(value - epsilon)
    return (loss_on_right - loss_on_left) / (2 * epsilon)


def gradient_descent(xs, targets, initial_weight, bias, learning_rate, steps):
    """Tek weight'i sayısal türev ve gradient descent ile öğren.

    Güncelleme kuralı:
        new_weight = old_weight - learning_rate * gradient

    Gradient loss'un artış yönünü gösterir. Eksi işareti sayesinde ters yöne,
    yani loss'un azaldığı yöne gideriz. Learning rate ise adım büyüklüğüdür.
    """
    weight = initial_weight
    history = []

    for step in range(steps + 1):
        # 1) FORWARD PASS + LOSS
        # Mevcut weight ile veri kümesindeki tahminlerin hatasını ölç.
        loss = dataset_loss(xs, targets, weight, bias)

        # Eğitimi sonradan inceleyebilmek ve CSV'ye yazabilmek için her adımın
        # parametresini ve loss'unu çalışma günlüğünde sakla.
        history.append((step, weight, loss))
        if step == steps:
            break

        # 2) GRADIENT
        # numerical_derivative tek argümanlı fonksiyon bekliyor. Bu küçük
        # fonksiyonda xs, targets ve bias sabit; yalnızca weight değişkendir.
        loss_for_weight = lambda candidate: dataset_loss(
            xs, targets, candidate, bias
        )
        gradient = numerical_derivative(loss_for_weight, weight)

        # 3) PARAMETRE GÜNCELLEMESİ
        # Modelin "öğrenmesi" bu satırdır: weight, loss'u azaltacak yönde
        # küçük bir adımla değiştirilir.
        weight = weight - learning_rate * gradient

    return weight, history
