# In diesem File kümmern wir uns um die Vorhersage der Farben der Büroklammer. Wir entwerfen dieses Modul so,
# dass es für jede detectierte Büroklammer aufgerufen wird und die Farbe der Büroklammer zurückgibt.
# Durch folgende Schritte wird die Farbe der Büroklammer bestimmt:
# ~ Zuerst werden die Kanten der Büroklammer mit der Canny-Kantenerkennung erkannt.
# ~ Dann wird der Kontrast des Bildes erhöht, indem nur die Kanten des Bildes maskiert werden.
# ~ Schließlich wird die Farbe der Büroklammer anhand der Farbe der maskierten Kanten klassifiziert.
# Am Ende wird die Farbe der Büroklammer an das Hauptprogramm zurückgegeben.

import cv2
import numpy as np
from matplotlib import pyplot as plt

def detect_edges(image_path, low_threshold=250, high_threshold=350):
    # Bild laden
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Verwendung der Canny-Kantenerkennung, um die Büroklammer zu erkennen
    edges = cv2.Canny(gray, low_threshold, high_threshold)

    # Durch Verwendung des 4x4-Kernels werden die Kanten erweitert, um mehr Fläche um die erkannten Kanten einzuschließen
    kernel = np.ones((4, 4), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    return edges, image


# In dieser Funktion kümmern wir uns um die farbliche Bearbeitung des Bildes.
def enhance_contrast(image, edges):
    '''
    Erhöht den Kontrast der Bereiche im Bild, die durch die Kantenmaske definiert sind.

    Diese Funktion nimmt ein Bild und eine Kantenmaske als Eingabe. Sie maskiert das Bild mit den Kanten,
    so dass nur die Bereiche, die durch die Kantenmaske definiert sind, bearbeitet werden. Anschließend wird
    der Kontrast in diesen maskierten Bereichen erhöht, um die Erkennung der Farben in diesen Bereichen zu verbessern.

    Parameter:
    ----------
    image : numpy.ndarray
        Das Eingabebild, bei dem der Kontrast erhöht werden soll. Es sollte ein farbiges Bild im BGR-Format sein.
    edges : numpy.ndarray
        Die Kantenmaske des Bildes, die die Bereiche definiert, in denen der Kontrast erhöht werden soll.
        Diese Maske sollte ein binäres Bild sein, bei dem die Kanten als nicht-null (nicht-schwarz) markiert sind.

    Rückgabewert:
    -------------
    enhanced_image : numpy.ndarray
        Das Bild mit erhöhtem Kontrast in den durch die Kantenmaske definierten Bereichen.
        Der Kontrast wird durch Skalierung der Pixelwerte in diesen Bereichen erhöht.

    '''
    # Als erstes wird das Originalbild mit den Kanten maskiert
    mask = np.zeros_like(image)
    mask[edges != 0] = image[edges != 0]

    # Anschließend wird der Kontrast nur im maskierten Bereich erhöht. Wahlweise kann auch die Helligkeit angepasst werden.
    # alpha = 2 erhöht den Kontrast, beta = 0 erhöht die Helligkeit
    enhanced_image = image.copy()
    enhanced_image[edges != 0] = cv2.convertScaleAbs(image[edges != 0], alpha=2, beta=5000)

    return enhanced_image


def classify_color(original_image, edges):
    '''
    Klassifiziert die Farbe der Bereiche im Bild, die durch die Kantenmaske definiert sind.

    Diese Funktion nimmt ein Bild und eine Kantenmaske als Eingabe. Sie erstellt eine Maske basierend auf den Kanten
    und extrahiert die Pixel des Originalbildes, die innerhalb der Kanten liegen. Anschließend werden die extrahierten
    Pixel in den HSV-Farbraum konvertiert und einer vordefinierten Farbpalette zugeordnet, um die dominierende Farbe
    zu bestimmen.

    Parameter:
    ----------
    original_image : numpy.ndarray
        Das Eingabebild, aus dem die Farben klassifiziert werden sollen. Es sollte ein farbiges Bild im BGR-Format sein.
    edges : numpy.ndarray
        Die Kantenmaske des Bildes, die die Bereiche definiert, deren Farben klassifiziert werden sollen.
        Diese Maske sollte ein binäres Bild sein, bei dem die Kanten als nicht-null (nicht-schwarz) markiert sind.

    Rückgabewert:
    -------------
    detected_color : str
        Die erkannte Farbe, die in den durch die Kantenmaske definierten Bereichen am häufigsten vorkommt.
        Die möglichen Rückgabewerte sind: 'black', 'white', 'blue', 'green', 'yellow', 'red', oder 'undefined',
        wenn keine Farbe erkannt werden konnte.
'''
    # Erstellen einer Maske mit den Kanten
    mask = edges != 0

    # Extrahieren der Pixel des orignial Bildes innerhalb der Kanten
    masked_image = original_image[mask]

    # Konvertieren der extrahierten Pixel in den HSV-Farbraum
    if masked_image.size == 0:
        return "undefined"
    hsv_image = cv2.cvtColor(masked_image.reshape(1, -1, 3), cv2.COLOR_BGR2HSV)

    # Die Farbbereiche im HSV-Farbraum haben wir so definiert, dass sie den Farben der Büroklammern entsprechen
    color_ranges = {
        'black': ([0, 0, 0], [180, 255, 50]),
        'white': ([0, 0, 200], [180, 20, 255]),
        'blue': ([100, 150, 0], [140, 255, 255]),
        'green': ([40, 50, 20], [90, 255, 255]),
        'yellow': ([20, 100, 100], [30, 255, 255]),
        'red': ([0, 100, 100], [10, 255, 255])
    }

    # Initialisieren der Farbzählungen
    color_counts = {color: 0 for color in color_ranges}

    # Klassifizieren der Farben in den extrahierten Pixeln
    for color, (lower, upper) in color_ranges.items():
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        mask = cv2.inRange(hsv_image, lower, upper)
        color_counts[color] = cv2.countNonZero(mask)

    # Nachdem die Farben klassifiziert wurden, wird die am häufigsten vorkommende Farbe bestimmt
    detected_color = max(color_counts, key=color_counts.get)

    return detected_color

# Diese Funktion ist der Einstiegspunkt für das Farberkennungsmodul.
def canny_hsv_classifier(image_path, low_threshold=250, high_threshold=350):
    '''
    Anwendung der Canny-Kantenerkennung und HSV-Farberkennung auf ein Bild zur Erkennung der dominanten Farbe.

    Diese Funktion kombiniert die Canny-Kantenerkennung und die HSV-Farberkennung, um die Farbe eines Objekts
    (in diesem Fall einer Büroklammer) im Bild zu bestimmen. Zuerst werden die Kanten des Objekts erkannt,
    dann wird der Kontrast des Bildes in den durch die Kanten definierten Bereichen erhöht, und schließlich
    wird die Farbe des Objekts klassifiziert.

    Parameter:
    ----------
    image_path : str
        Der Pfad zum Eingabebild, das analysiert werden soll.
    low_threshold : int, optional
        Der untere Schwellenwert für die Canny-Kantenerkennung. Standard ist 250.
    high_threshold : int, optional
        Der obere Schwellenwert für die Canny-Kantenerkennung. Standard ist 350.

    Rückgabewert:
    -------------
    edges : numpy.ndarray
        Das binäre Bild, das die durch die Canny-Kantenerkennung erkannten Kanten zeigt.
    enhanced_image : numpy.ndarray
        Das Bild, bei dem der Kontrast in den durch die Kanten definierten Bereichen erhöht wurde.
    detected_color : str
        Die erkannte Farbe, die in den durch die Kantenmaske definierten Bereichen am häufigsten vorkommt.
        Die möglichen Rückgabewerte sind: 'black', 'white', 'blue', 'green', 'yellow', 'red', oder 'undefined',
        wenn keine Farbe erkannt werden konnte.
    '''
    edges, original_image = detect_edges(image_path, low_threshold, high_threshold)
    enhanced_image = enhance_contrast(original_image, edges)
    detected_color = classify_color(original_image, edges)

    print(f"Detected color: {detected_color}")

    return detected_color

# Für Debugging-Zwecke kann dieses Programm auch direkt ausgeführt werden.
if __name__ == "__main__":
    image_path = r"A:\Paperclip Models\single_paperclips\predictions\test_3.jpg"
    detected_color = canny_hsv_classifier(image_path)

    # Plot the images
    edges, original_image = detect_edges(image_path)
    enhanced_image = enhance_contrast(original_image, edges)

    plt.subplot(131), plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(edges, cmap='gray')
    plt.title('Detected Edges'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
    plt.title('Enhanced Image'), plt.xticks([]), plt.yticks([])
    plt.show()

