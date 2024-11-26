import numpy as np
from PIL import Image


def value_to_rgb(value, min_value, max_value):
    normalized_value = (value - min_value) / (max_value - min_value)

    if normalized_value < 0.25:
        # Schwarz -> Blau
        r = 0
        g = 0
        b = int(normalized_value * 4 * 255)  # Blau steigt linear
    elif normalized_value < 0.5:
        # Blau -> Rot
        r = int((normalized_value - 0.25) * 4 * 255)  # Rot steigt linear
        g = 0
        b = 255 - int((normalized_value - 0.25) * 4 * 255)  # Blau fällt linear
    elif normalized_value < 0.75:
        # Rot -> Gelb
        r = 255
        g = int((normalized_value - 0.5) * 4 * 255)  # Grün steigt linear
        b = 0
    else:
        # Gelb -> Weiß
        r = 255
        g = 255
        b = int((normalized_value - 0.75) * 4 * 255)  # Blau steigt linear

        # Rückgabe des RGB-Werts als Tupel
    return (r, g, b)


with open('../TIR_imgs/m201017020400594_336x252_14bit.thermal.celsius.csv', 'r') as file:
    # Lesen des gesamten Inhalts der Datei
    head, inhalt = file.read().split("\n\n")

    rows = inhalt.strip().split("\n")

    data = []
    for row in rows:
        data.append([])
        for value in row.split(";"):
            if value.strip() != "":
                data[-1].append(float(value))

    min_value = np.min(data)
    max_value = np.max(data)

    data = np.array(data)

    # Normalisieren der Daten
    # data -= min_value
    # data /= (max_value - min_value)
    # data *= 255

    # Jetzt erstelle ein RGB-Bild:
    # Wir erstellen ein 3D-Array mit den gleichen Dimensionen wie das Originalbild,
    # aber mit 3 Kanälen für RGB.
    rgb_data = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.uint8)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # Setze den gleichen Wert für alle drei Kanäle (RGB)
            rgb_data[i, j] = value_to_rgb(data[i, j], min_value, max_value)

    # Convert the NumPy array to a PIL Image (RGB)
    image = Image.fromarray(rgb_data)

    # Display the image
    image.show()

    # Optionally, save the image
    # image.save('output_image.png')
