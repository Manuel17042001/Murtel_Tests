from datetime import timedelta, datetime

import cv2
import numpy as np
import csv
import os


def csv_to_color_image(csv_path):
    """
    Liest eine CSV-Datei mit Temperaturwerten und wandelt sie in ein Falschfarbenbild um.
    """
    try:
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            data = []

            for row in reader:
                try:
                    # Überprüfe, ob die Zeile numerische Daten enthält
                    data_row = [float(val) for val in row if val.strip()]
                    data.append(data_row)
                except ValueError:
                    continue  # Ignoriere Header-Zeilen

        # Konvertiere die Daten in ein NumPy-Array
        data_array = np.array(data)

        # Normalisiere die Temperaturwerte auf einen Bereich von 0-255
        min_val, max_val = np.min(data_array), np.max(data_array)
        normalized_data = ((data_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)

        # Wende ein Colormap an, um das Bild in Farben darzustellen
        color_image = cv2.applyColorMap(normalized_data, cv2.COLORMAP_JET)

        return color_image
    except UnicodeDecodeError as e:
        print(f"Fehler beim Lesen der Datei {csv_path}: {e}")
        return None



def align_images(rgb_image_path, csv_path, output_path):
    # Lade das RGB-Bild
    rgb_image = cv2.imread(rgb_image_path)
    if rgb_image is None:
        print(f"Fehler beim Laden des RGB-Bildes: {rgb_image_path}")
        return

    # Konvertiere die CSV-Daten in ein Falschfarbenbild
    ir_image_color = csv_to_color_image(csv_path)
    if ir_image_color is None:
        print(f"Fehler beim Erstellen des Falschfarbenbildes: {csv_path}")
        return

    # Konvertiere das RGB-Bild in Graustufen für die Merkmalserkennung
    gray_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)


    # Initialisiere den ORB-Detektor (Merkmalserkennung)
    orb = cv2.ORB_create()

    # Finde Merkmale und Keypoints
    keypoints1, descriptors1 = orb.detectAndCompute(gray_rgb, None)
    keypoints2, descriptors2 = orb.detectAndCompute(cv2.cvtColor(ir_image_color, cv2.COLOR_BGR2GRAY), None)

    # Matcher für Keypoints (z.B. Brute Force Matcher)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sortiere Matches nach Qualität
    matches = sorted(matches, key=lambda x: x.distance)

    # Verwende die besten Matches
    num_good_matches = min(50, len(matches))  # Passe dies ggf. an
    good_matches = matches[:num_good_matches]

    # Extrahiere die Punktpaare
    points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
    points2 = np.zeros((len(good_matches), 2), dtype=np.float32)

    for i, match in enumerate(good_matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Berechne die Homographie-Matrix
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

    # Wende die Transformation auf das Infrarotbild an
    height, width, _ = rgb_image.shape
    aligned_ir_color = cv2.warpPerspective(ir_image_color, h, (width, height))

    # Speichere das ausgerichtete Bild
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, os.path.basename(csv_path).replace('.csv', '_aligned.png'))
    cv2.imwrite(output_file, aligned_ir_color)

    print(f"Gespeichertes ausgerichtetes Bild: {output_file}")


def find_files_with_prefix(directory, prefix):
    """
    Sucht nach Dateien in einem Verzeichnis, die mit einem bestimmten Präfix beginnen.

    :param directory: Verzeichnis, in dem gesucht werden soll
    :param prefix: Präfix, nach dem gesucht werden soll
    :return: Liste der passenden Dateien
    """
    matching_files = []
    try:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.startswith(prefix):
                    matching_files.append(os.path.join(root, file))
    except Exception as e:
        print(f"Fehler bei der Suche: {e}")
    return matching_files


def parse_image_time(filename):
    # Handle filenames without extensions
    try:
        if '.' in filename:
            base, ext = filename.split('.')
        else:
            base = filename

        if len(base) < 12:
            return None  # Skip invalid filenames

        year = int("20" + base[1:3])  # Add '20' to the year
        month = int(base[3:5])
        day = int(base[5:7])
        hour = int(base[7:9])
        minute = int(base[9:11])

        return datetime(year, month, day, hour, minute)
    except Exception as e:
        # print(f"Error parsing {filename}: {e}")
        return None


def find_nearest_image(target_filename, filenames, time_window=20):
    target_time = parse_image_time(target_filename)
    if target_time is None:
        return None

    nearest_image = None
    min_time_diff = timedelta(minutes=time_window)

    for filename in filenames:
        current_time = parse_image_time(filename.split("\\")[1])
        if current_time is None:
            continue

        time_diff = abs(target_time - current_time)
        if time_diff <= timedelta(minutes=time_window) and time_diff < min_time_diff:
            nearest_image = filename
            min_time_diff = time_diff

    return nearest_image


# Beispiel: Bearbeitung eines Datasets
def process_dataset(rgb_dir, csv_dir, output_dir):
    for rgb_file in os.listdir(rgb_dir):
        if rgb_file.endswith('.jpg'):
            rgb_path = os.path.join(rgb_dir, rgb_file)
            csv_filenames = find_files_with_prefix(csv_dir, prefix=rgb_file[0:5])
            csv_path = find_nearest_image(rgb_file, csv_filenames, time_window=20)
            print(rgb_file)
            print(csv_path)
            if csv_path is None:
                continue
            if os.path.exists(csv_path):
                align_images(rgb_path, csv_path, output_dir)


# Eingaben anpassen
rgb_directory = "../RGB_imgs"
csv_directory = "../TIR_imgs"
output_directory = "output_aligment"

process_dataset(rgb_directory, csv_directory, output_directory)
