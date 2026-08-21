"""YZ50 birinci hafta görevlerini sırayla çalıştıran deney defteri.

Bu dosya sonuçları yalnızca üretmez; her bölüm aynı zamanda temel bir neural
network kavramını gözle görünür hale getirir. Matematiksel fonksiyonların yalın
halleri ``neural_network.py`` dosyasındadır.
"""

import csv
from pathlib import Path

from neural_network import (
    dataset_loss,
    gradient_descent,
    layer_forward,
    mean_squared_error,
    neuron_forward,
)


OUTPUT_DIR = Path(__file__).parent / "outputs"


def write_history_csv(history, path):
    """Gradient descent adımlarını sonradan incelemek için CSV olarak kaydet."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "weight", "loss"])
        writer.writerows(history)


def write_loss_svg(points, path):
    """Üçüncü taraf çizim kütüphanesi olmadan basit bir loss grafiği üret."""
    # SVG metin tabanlı bir resim formatıdır. Aşağıdaki kod matematiğin parçası
    # değil; hesapladığımız (weight, loss) noktalarını görünür kılmak içindir.
    width, height, margin = 800, 450, 55
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)

    def scale_x(value):
        return margin + (value - x_min) / (x_max - x_min) * (width - 2 * margin)

    def scale_y(value):
        return height - margin - (value - y_min) / (y_max - y_min) * (height - 2 * margin)

    polyline = " ".join(
        f"{scale_x(x):.2f},{scale_y(y):.2f}" for x, y in points
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#222"/>
  <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#222"/>
  <polyline points="{polyline}" fill="none" stroke="#2563eb" stroke-width="3"/>
  <text x="{width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="20">Weight'e gore loss</text>
  <text x="{width/2}" y="{height-12}" text-anchor="middle" font-family="sans-serif">weight</text>
  <text x="16" y="{height/2}" text-anchor="middle" font-family="sans-serif" transform="rotate(-90 16 {height/2})">loss</text>
  <text x="{margin}" y="{height-margin+22}" font-family="sans-serif" font-size="12">{x_min:.1f}</text>
  <text x="{width-margin}" y="{height-margin+22}" text-anchor="end" font-family="sans-serif" font-size="12">{x_max:.1f}</text>
  <text x="{margin-8}" y="{margin+4}" text-anchor="end" font-family="sans-serif" font-size="12">{y_max:.2f}</text>
  <text x="{margin-8}" y="{height-margin}" text-anchor="end" font-family="sans-serif" font-size="12">{y_min:.2f}</text>
</svg>'''
    path.write_text(svg, encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ---------------------------------------------------------------------
    # GÖREV 1 — TEK NÖRON FORWARD PASS
    # ---------------------------------------------------------------------
    # Hesap: 1*0.2 + 2*0.8 + 3*(-0.5) + 2 = 2.3
    # inputs veridir; weights ve bias modelin parametreleridir.
    print("1) Tek noron forward pass")
    output = neuron_forward([1.0, 2.0, 3.0], [0.2, 0.8, -0.5], 2.0)
    print(f"   output = {output:.4f}")

    # ---------------------------------------------------------------------
    # GÖREV 2 — BİR KATMANIN FORWARD PASS'İ
    # ---------------------------------------------------------------------
    # Üç nöron aynı girdileri alıyor. Her satır farklı bir nöronun weight
    # listesidir ve her nöronun kendine ait bias değeri vardır.
    print("\n2) Uc noronlu katman forward pass")
    outputs = layer_forward(
        [1.0, 2.0, 3.0],
        [[0.2, 0.8, -0.5], [0.5, -0.91, 0.26], [-0.26, -0.27, 0.17]],
        [2.0, 3.0, 0.5],
    )
    print(f"   outputs = {[round(value, 4) for value in outputs]}")

    # ---------------------------------------------------------------------
    # GÖREV 3 — LOSS FONKSİYONU
    # ---------------------------------------------------------------------
    # Loss tahminin ne kadar kötü olduğunu tek sayıya indirger. Buradaki MSE:
    # [(2.5-3)^2 + (0-(-0.5))^2 + (2.1-2)^2] / 3 = 0.17
    print("\n3) Mean squared error")
    loss = mean_squared_error([2.5, 0.0, 2.1], [3.0, -0.5, 2.0])
    print(f"   loss = {loss:.4f}")

    # Son iki görev için küçük ve ilişkisi bilinen bir veri kümesi kullanılır.
    # Gerçek kural y=2x'tir. Model bu kuralı önceden bilmiyor; doğru weight'i
    # loss'a bakarak bulmasını istiyoruz.
    xs = [-2.0, -1.0, 0.0, 1.0, 2.0]
    targets = [-4.0, -2.0, 0.0, 2.0, 4.0]  # Gerçek ilişki: y = 2x
    bias = 0.0

    # ---------------------------------------------------------------------
    # GÖREV 4 — PARAMETREYİ ELLE DEĞİŞTİRMEK
    # ---------------------------------------------------------------------
    # Weight'i -1 ile 3 arasında tarıyoruz. Bu bir eğitim algoritması değil;
    # parametre-loss ilişkisini gözlemlemek için kontrollü bir deneydir.
    # Weight 2 olduğunda model y=2x kuralıyla aynı olur ve loss sıfıra iner.
    print("\n4) Parametreyi manuel degistirme")
    curve = []
    for index in range(21):
        weight = -1.0 + index * 0.2  # Her denemede weight'i 0.2 artır.
        current_loss = dataset_loss(xs, targets, weight, bias)
        curve.append((weight, current_loss))
        print(f"   weight={weight:5.2f} -> loss={current_loss:7.4f}")
    # Hesaplanan noktaları outputs/loss_curve.svg içinde görselleştir.
    write_loss_svg(curve, OUTPUT_DIR / "loss_curve.svg")

    # ---------------------------------------------------------------------
    # GÖREV 5 — SAYISAL TÜREVLE GRADIENT DESCENT
    # ---------------------------------------------------------------------
    # Bu kez doğru weight'i elle seçmiyoruz. Başlangıçta -1 olan weight,
    # sayısal türevin gösterdiği azalış yönünde 20 kez güncelleniyor.
    # Beklenti: weight 2'ye yaklaşırken loss'un sıfıra yaklaşmasıdır.
    print("\n5) Sayisal turev ile gradient descent")
    final_weight, history = gradient_descent(
        xs, targets, initial_weight=-1.0, bias=bias, learning_rate=0.1, steps=20
    )
    for step, weight, current_loss in history:
        print(f"   step={step:2d}, weight={weight:9.6f}, loss={current_loss:.8f}")

    # CSV dosyası, öğrenme sürecindeki her adımı tablo olarak saklar.
    write_history_csv(history, OUTPUT_DIR / "gradient_descent.csv")
    print(f"\nFinal weight: {final_weight:.6f}")
    print(f"Ciktilar: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
