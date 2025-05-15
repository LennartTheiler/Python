# In diesem Projekt baue ich eine Objekterkennung, die Flugzeuge am Himmel erkennt, trackt und zählen kann.
# Die Rohdatei habe ich aus diesem Video: https://www.youtube.com/watch?v=RBIpp3B-U9k&t=115s
# Ich nutze Yolov8 und OpenCV für die Objekterkennung

import cv2
import torch
from ultralytics import YOLO

# Als erstes überprüfe ich ob ich Cuda richtig installiert habe, damit ich die Berechnungen auf meiner Grafikkarte
# durchführen kann.
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"CUDA verfügbar! GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')  # Fallback auf CPU, wenn CUDA nicht verfügbar ist
    print("CUDA nicht verfügbar, auf CPU umgeschaltet.")

# ----------------------------------------------------------------------------------------------------------------------
# 1. Model laden
# ----------------------------------------------------------------------------------------------------------------------
# Ich nutze hier ein vortrainiertes Modell (Yolov8)
model = YOLO('yolov8n.pt')

# ----------------------------------------------------------------------------------------------------------------------
# 2. Video laden & vorverarbeiten
# ----------------------------------------------------------------------------------------------------------------------
input_path = r'C:\Users\lenna\Desktop\Studium\Semester 5\Rheinmetall\projekt_rheinmetall\Rohdatei.mp4'
output_path = r'C:\Users\lenna\Desktop\Studium\Semester 5\Rheinmetall\projekt_rheinmetall\output1.mp4'

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError(f"Konnte Video nicht öffnen: {input_path}")

# Nachdem das Video geladen ist, sorge ich jetzt dafür, dass es in die richtige Auflösung umgewandelt wird
fps = cap.get(cv2.CAP_PROP_FPS)
width, height = 1280, 720
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# ----------------------------------------------------------------------------------------------------------------------
# 3. Frame-Loop: Erkennung, Tracking & Visualisierung
# ----------------------------------------------------------------------------------------------------------------------
# Da ich die Erkennung und das Tracking in jedem Frame durchführen muss, führe ich eine Schleife ein, die dies für mich
# erledigt. Es wird als erstes der nächste Frame geladen und noch mal auf 720p skaliert, falls nötig
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (width, height))

# ----------------------------------------------------------------------------------------------------------------------
# 4. Objekterkennung und Tracking
# ----------------------------------------------------------------------------------------------------------------------
# Das Konfidenzniveau für die Erkennung lege ich auf 0.25 und den Schwellenwert für die Overlap-Kontrolle auf 0.45
# Diese Werte haben sich in meinem Hyperparametertuning ergeben
    results = model.track(frame, conf=0.25, iou=0.45, verbose=False)

    # Zähle die Anzahl der Flugzeuge
    num_airplanes = 0

    # Visualisierung: Zeichnen von Bounding Boxes und Track-IDs für erkannte Objekt
    # Ich gehe hier durch alle erkannten Objekte in einem Frame und alle Bounding Boxen im aktuellen Ergebnis
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            # Hier hole ich die Klassen der Objekte. In der aktuellen Version sind das nur Flugzeuge
            name = model.names[cls]
            # Wenn das erkannte Objekt ein Flugzeug ist, zeichne ich die Bounding Box
            if name.lower() == 'airplane' or name.lower() == 'aeroplane':
                # Um die Bounding Box zeichnen zu können, hole ich erst die Koordinaten (x1, y1, x2, y2) und dann
                # die Track-ID
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id[0])
                # Ab hier zeichne ich die Bounding Box, Objektnamen und die Track-ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{name}-{track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                # Wenn ich nun ein Flugzeug erkannt habe, zähle ich den Counter hoch
                num_airplanes += 1

    # Nun wird die Anzahl der gesamt gezählten Flugzeuge auf dem Video platziert
    cv2.putText(frame, f"Flugzeuge gefunden: {num_airplanes}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    out.write(frame)

cap.release()
out.release()
# Ausgabe, dass das Tracking abgeschlossen ist und die Datei gespeichert wurde
print(f"Tracking abgeschlossen. Output gespeichert in: {output_path}")
