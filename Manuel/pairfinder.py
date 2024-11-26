from datetime import timedelta, datetime
import os


def find_csv_files_with_prefix(directory, prefix):
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
                if file.endswith(".csv"):
                    if file.startswith(prefix):
                        matching_files.append(os.path.join(root, file))
    except Exception as e:
        print(f"Fehler bei der Suche: {e}")
    return matching_files


def parse_image_time(filename):
    """
    Extrahiert die Zeitinformationen aus einem Dateinamen im erwarteten Format.

    :param filename: Dateiname, aus dem Zeitinformationen extrahiert werden sollen.
    :return: Datetime-Objekt oder None, wenn der Name ungültig ist.
    """
    try:
        if len(filename) < 12:
            return None  # Überspringe ungültige Dateinamen

        year = int("20" + filename[1:3])  # Addiere '20' zum Jahr
        month = int(filename[3:5])
        day = int(filename[5:7])
        hour = int(filename[7:9])
        minute = int(filename[9:11])
        return datetime(year, month, day, hour, minute)
    except Exception as e:
        print(f"Error by parsing image to time: {e}")
        return None


def find_nearest_image(target_filename, filenames, time_window=20):
    """
    Findet das nächste Bild basierend auf Zeitinformationen.

    :param target_filename: Ziel-RGB-Dateiname.
    :param filenames: Liste von Dateinamen zur Suche.
    :param time_window: Maximale Zeitdifferenz (in Minuten) für ein passendes Paar.
    :return: Pfad zum nächsten Bild oder None.
    """
    target_time = parse_image_time(target_filename)
    if target_time is None:
        return None

    nearest_image = None
    min_time_diff = timedelta(minutes=time_window)

    for filename in filenames:
        current_time = parse_image_time(os.path.basename(filename))
        if current_time is None:
            continue

        time_diff = abs(target_time - current_time)
        if time_diff <= timedelta(minutes=time_window) and time_diff < min_time_diff:
            nearest_image = filename
            min_time_diff = time_diff

    return nearest_image


def save_pairs_to_file(pairs, output_file):
    """
    Speichert die gefundenen Bildpaare in eine Datei.

    :param pairs: Liste von Tupeln (RGB-Bild, TIR-Bild).
    :param output_file: Pfad zur Ausgabedatei.
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            for rgb, tir in pairs:
                file.write(f"{rgb};{tir}\n")
        print(f"Paare erfolgreich gespeichert in {output_file}")
    except Exception as e:
        print(f"Fehler beim Speichern der Paare: {e}")


def find_image_pairs(rgb_dir, csv_dir, output_file, time_window=20):
    """
    Findet und speichert Bildpaare basierend auf Zeitinformationen.

    :param rgb_dir: Verzeichnis mit RGB-Bildern.
    :param csv_dir: Verzeichnis mit CSV-Dateien (Infrarot).
    :param output_file: Pfad zur Ausgabedatei, in der die Paare gespeichert werden.
    :param time_window: Zeitfenster in Minuten.
    """
    pairs = []
    for rgb_file in os.listdir(rgb_dir):
        if rgb_file.endswith('.jpg'):
            rgb_path = os.path.join(rgb_dir, rgb_file)
            csv_filenames = find_csv_files_with_prefix(csv_dir, prefix=rgb_file[0:5])
            csv_path = find_nearest_image(rgb_file, csv_filenames, time_window)
            if csv_path is not None:
                pairs.append((rgb_path, csv_path))

    # Speichere die Paare in einer Datei
    save_pairs_to_file(pairs, output_file)


# Beispielhafte Verwendung
if __name__ == "__main__":
    rgb_directory = "../RGB_imgs"
    csv_directory = "../TIR_imgs"
    output_file = "image_pairs.txt"
    find_image_pairs(rgb_directory, csv_directory, output_file, time_window=20)
