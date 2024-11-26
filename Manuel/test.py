import cv2
import numpy as np
from matplotlib import pyplot as plt


def analyze_image(image_path):
    # Lade das Bild
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Bild konnte nicht geladen werden: {image_path}")

    # Konvertiere zu Graustufen
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Kantenextraktion mit Canny-Detektor
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    # Berechnung der Farbvarianz (Varianz der RGB-Kanäle)
    variance = np.var(img, axis=(0, 1))

    # Summiere die Anzahl der Kantenpixel
    edge_density = np.sum(edges) / edges.size

    # Zeige Ergebnisse
    print(f"Farbvarianz (RGB): {np.mean(variance)}")
    print(f"Kantendichte: {edge_density:.6f}")

    # Entscheidung basierend auf Farbvarianz und Kantendichte
    if edge_density < 0.05 and np.mean(variance) < 2000:
        result = "Wahrscheinlich Nebel"
    elif edge_density >= 0.05 and np.mean(variance) >= 2000:
        result = "Wahrscheinlich Schnee"
    else:
        result = "Unklar"

    print(f"Ergebnis: {result}")

    # Zeige das Bild und die Kanten
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Originalbild")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Kanten (Canny)")
    plt.imshow(edges, cmap='gray')
    plt.axis("off")

    plt.show()
    return result


# Testen mit einem Bild
image_path = "../RGB_imgs/m201014200425513.jpg"
result = analyze_image(image_path)
