# Flugzeugerkennung und -tracking mit YOLOv8

In diesem Projekt wird ein Deep-Learning-basiertes System entwickelt, das Flugzeuge am Himmel in einem Video erkennt, verfolgt (Tracking) und zählt. Ziel ist es, mithilfe eines vortrainierten YOLOv8-Modells automatisch Flugzeuge in Videomaterial zu identifizieren und die Ergebnisse visuell aufzubereiten.

## Inhalt des Projekts

- `projekt_rheinmetall.py`: Das Hauptskript des Projekts. Es führt folgende Schritte aus:
  - Lädt ein Eingabevideo
  - Verwendet YOLOv8 zur Erkennung von Flugzeugen in jedem Frame
  - Verfolgt erkannte Flugzeuge über mehrere Frames hinweg
  - Zeichnet Bounding Boxes und Track-IDs ins Video ein
  - Gibt die Gesamtanzahl erkannter Flugzeuge im Bild aus
  - Speichert das Ergebnisvideo

## Verwendete Technologien

- [YOLOv8](https://github.com/ultralytics/ultralytics): Ein modernes Echtzeit-Objekterkennungsmodell
- [OpenCV](https://opencv.org/): Für Videobearbeitung und Visualisierung
- [PyTorch](https://pytorch.org/): Für die Nutzung der GPU und das Backend von YOLO

## Videos

Im Projektordner befinden sich zwei Videos:

- `original_video.mp4` *(Name ggf. anpassen)*: Das unbearbeitete Eingabevideo mit Flugzeugen am Himmel
- `output_video.mp4` *(Name ggf. anpassen)*: Das von YOLOv8 verarbeitete Video mit eingezeichneten Bounding Boxes, Track-IDs und Flugzeugzähler

## Voraussetzungen

- Python 3.8+
- Installierte Pakete:
  ```bash
  pip install ultralytics opencv-python torch
