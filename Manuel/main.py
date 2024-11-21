import cv2
import numpy as np
import os
import shutil
from tqdm import tqdm  # Für die Fortschrittsanzeige


def analyze_image(image_path):
    """
    Analysiert ein Bild, um zwischen Nebel und Schnee zu unterscheiden.
    """
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

    # Entscheidung basierend auf Farbvarianz und Kantendichte
    if edge_density < 30 and np.mean(variance) < 2000 and np.mean(variance) > 400:
        return "Nicht-Nebel"
    else:
        return "Nebel"


def process_images(input_folder, fog_folder, not_fog_folder):
    """
    Verarbeitet alle Bilder in einem Ordner und verschiebt jene mit Nebel in einen Zielordner.
    Zeigt den Fortschritt mit einer Progressbar an.
    """
    if not os.path.exists(fog_folder):
        os.makedirs(fog_folder)

    if not os.path.exists(not_fog_folder):
        os.makedirs(not_fog_folder)

    # Liste aller Bilddateien im Eingabeordner
    image_files = [
        file_name for file_name in os.listdir(input_folder)
        if os.path.isfile(os.path.join(input_folder, file_name)) and file_name.lower().endswith(
            ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    ]

    # Fortschrittsanzeige initialisieren
    for file_name in tqdm(image_files, desc="Bilder verarbeiten", unit="Bild"):
        file_path = os.path.join(input_folder, file_name)
        try:
            # Analyse des Bildes
            result = analyze_image(file_path)

            # Falls Nebel erkannt wird, kopiere das Bild in den Zielordner
            if result == "Nebel":
                shutil.copy(file_path, os.path.join(fog_folder, file_name))
            else:
                shutil.copy(file_path, os.path.join(not_fog_folder, file_name))
        except Exception as e:
            print(f"Fehler bei der Verarbeitung von {file_name}: {e}")


# Ordnerpfade
input_folder = "RGB_imgs/RGB_imgs"  # Ersetze durch den Pfad zu deinem Eingabeordner
fog_folder = "not_useful"  # Ersetze durch den Pfad zum Zielordner
not_fog_folder = "useful"

# Verarbeitung starten
process_images(input_folder, fog_folder, not_fog_folder)
