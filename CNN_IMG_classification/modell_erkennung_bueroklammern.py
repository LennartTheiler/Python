# Dieses Skript wurde erstellt von:
# ~ Lennart Theiler ~ Matrikelnummer: 90878
# ~ David Pfliehinger ~ Matrikelnummer: 91913

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import xml.etree.ElementTree as ET
import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import color_detection

# Hier wird überprüft, ob eine GPU verfügbar ist. Wenn eine GPU verfügbar ist, wird sie verwendet, andernfalls wird die
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------------------------------------------------------------------------------------------------
# Define the PaperclipDataset class
class PaperclipDataset(Dataset):
    '''Erstellen eines "target" Dictionaries, das die Bounding Boxes und Labels für ein bestimmtes Bild speichert
           und zusammen mit dem zugehörigen Bild ausgibt.'''
    def __init__(self, img_files, img_dir, annotations_dir, transform=None):
        self.img_files = img_files
        self.img_dir = img_dir
        self.annotations_dir = annotations_dir
        self.transform = transform

    # Wird Automatisch bei der Initialisierung des Datasets aufgerufen und gibt die Anzahl der Bilder zurück
    def __len__(self):
        return len(self.img_files)

    # Wird durch den DataLoader aufgerufen, um Bounding Boxes und Labels für ein bestimmtes Bild zu erhalten
    # und die Spezifikationen in einem Dictionary zurückzugeben
    def __getitem__(self, idx):
        '''Gibt ein Bild und die zugehörigen Bounding Boxes und Labels als dictionary zurück'''
        # Zusammensetzen des Pfades zum Bild und zur Annotation
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        annotation_path = os.path.join(self.annotations_dir, self.img_files[idx].replace('.jpg', '.xml'))

        # Öffnen des Bildes und konvertieren in RGB um sicherzustellen, dass es 3 Kanäle hat
        image = Image.open(img_path).convert("RGB")
        # Parsen der XML-Datei, um Bounding Boxes und Labels zu erhalten
        boxes, labels = self.parse_xml(annotation_path)

        if self.transform:
            # Anwenden der Transformationen auf das Bild
            image = self.transform(image)

        # Die Liste der Bounding Boxes und das label als Tensor in einem Dictionary speichern
        target = {}
        # Bounding Boxes als Float-Tensor mit 32 Bit genauigkeit speichern
        target['boxes'] = torch.tensor(boxes, dtype=torch.float32)
        # Bounding Boxes als Int-Tensor mit 64 Bit genauigkeit speichern
        target['labels'] = torch.tensor(labels, dtype=torch.int64)

        return image, target

    def parse_xml(self, xml_path):
        '''Position der Bounding Box auslesen und Label setzen'''
        # Parsen der XML-Datei
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes = []
        labels = []

        # Iterieren über alle Objekte in der XML-Datei.
        # Es wird nach dem Ausdruck '<object>' innerhalb der xml-Datei gesucht.
        # Nach '<object>' befindet sich die Bounding Box, die die Position des Objekts im Bild angibt.
        # Die Position der Bounding Box wird in Form von Bildkoordinaten in xmin, ymin, xmax, ymax gespeichert.
        for obj in root.findall('object'):
            label = 1  # Annahme: Label 1 für Büroklammern
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        return boxes, labels

# ----------------------------------------------------------------------------------------------------------------------
class DataHandler:
    '''Erstellen von Trainings- und Validierungsdaten für das Modell'''
    def __init__(self, img_dir, annotations_dir, val_ratio=0.2, batch_size=32, transform=None):
        self.img_dir = img_dir
        self.annotations_dir = annotations_dir
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.transform = transform

    def train_val_split(self):
        # Erstellen einer Liste aller Bilddateien innerhalb des Verzeichnisses mit den Trainingsdateien
        img_files = [f for f in os.listdir(self.img_dir) if f.endswith('.jpg')]
        # Überprüfen, ob für jedes Bild eine XML-Datei vorhanden ist
        img_files = [f for f in img_files if os.path.exists(os.path.join(self.annotations_dir, f.replace('.jpg', '.xml')))]
        random.shuffle(img_files) # Zufälliges Mischen der Bilddateien
        val_split = int(len(img_files) * self.val_ratio) # Berechnen der länge des Validierungsdatensatzes
        val_files = img_files[:val_split]
        train_files = img_files[val_split:]
        return train_files, val_files

    def create_dataloaders(self):
        '''Erstellen von Trainings- und Validierungsdataloadern für das Modell'''
        # Erstellen des Trainings- und Validierungsdatensatzes
        train_files, val_files = self.train_val_split()

        # Erstellen von Trainings- und Validierungsdatensätzen
        train_dataset = PaperclipDataset(train_files, self.img_dir, self.annotations_dir, self.transform)
        val_dataset = PaperclipDataset(val_files, self.img_dir, self.annotations_dir, self.transform)

        # Erstellen von Trainings- und Validierungsdataloadern
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader

# ----------------------------------------------------------------------------------------------------------------------
# Hier haben wir ein CNN implementiert, welches die Klasse (1 oder 0 / Klammer oder Tisch) und die Bounding Boxes
# vorhersagt. Durch eingehende Recherche haben wir herausgefunden, dass es am besten ist, die Klassen und die Bounding
# Boxes gleichzeitig vorherzusagen, da dies die Genauigkeit des Modells erhöht.
# Um die Genauigkeit des Modells weiter zu erhöhen, haben wir uns aus folgenden Gründen für ein Transfer-Learning-Modell
# entschieden:
# ~ Unser Dataset ist relativ klein (ca. 750 Bilder)
# ~ Transfer-Learning ist eine gängige Praxis, um die Genauigkeit von Modellen zu erhöhen

# Wir haben verschiedene Backbones getestet:darunter EfficientNet, MobileNet und ResNet.
# ~ MobileNet: Hat kaum Verbesserung gebracht
# ~ ResNet: Hat die Trainingszeit drastisch erhöht
# ~ EfficientNet: Hat die Genauigkeit des Modells verbessert und die Trainingszeit war akzeptabel
# Somit haben wir uns für das EfficientNet entschieden.

# Definition des Modells
class EfficientNetObjectDetector(nn.Module):

    '''
    Ein neuronales Netzwerk-Modell für die Objekterkennung, basierend auf EfficientNet-B0.

    Attribute:
    ----------
    backbone : torchvision.models.efficientnet.EfficientNet
        Das vortrainierte EfficientNet-B0-Modell, das als Feature-Extractor dient.
    classifier : torch.nn.Sequential
        Eine Sequenz von voll verbundenen Schichten zur Klassifikation (Papierklammer oder Hintergrund).
    box_regressor : torch.nn.Sequential
        Eine Sequenz von voll verbundenen Schichten zur Vorhersage der Bounding-Box-Koordinaten.

    Methoden:
    --------
    forward(x)
        Führt eine Vorwärtsdurchlauf des Modells durch und gibt die Klassifikations- und Bounding-Box-Vorhersagen zurück.
    '''

    def __init__(self):
        super(EfficientNetObjectDetector, self).__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1).features
        # Hier wird als erstes ein Fully Connected Layer mit 512 Neuronen und ReLU-Aktivierungsfunktion definiert.
        # Welches vorhersagen soll, ob ein Paperclip oder der Hintergrund vorliegt.
        self.classifier = nn.Sequential(
            nn.Linear(1280 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 2)  # Binäre Klassifikation: Paperclip oder Hintergrund
        )
        # Hier wird ein  weiteres Fully Connected Layer mit 512 Neuronen und ReLU-Aktivierungsfunktion definiert.
        # Dieses soll die Bounding Box der Paperclips vorhersagen.
        self.box_regressor = nn.Sequential(
            nn.Linear(1280 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 4)  # Vorhersage der Bounding Box (x1, y1, x2, y2)
        )

    # Hier wird die forward-Methode definiert, welche die Eingabe durch das Backbone-Netzwerk leitet und die Ausgabe
    # der Fully Connected Layer zurückgibt.
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        classes = self.classifier(x)
        boxes = self.box_regressor(x)
        return classes, boxes

# ----------------------------------------------------------------------------------------------------------------------
# Hier wird der PaperclipTrainer definiert, der das Modell trainiert. Der Trainer verwendet die Cross-Entropy-Loss-Funktion
# für die Klassifikation und die Mean Squared Error-Loss-Funktion für die Regression der Bounding Box.
# Der Trainer führt das Training für eine bestimmte Anzahl von Epochen durch und gibt den Verlust für jede Epoche aus.
# Adam-Optimizer: Sorgt für eine adaptive Lernrate, die sich während des Trainings anpasst. -> Schnelleres Training
# Cross-Entropy-Loss: Wird verwendet, um die Klassifikation zu bewerten. -> Gibt den Fehler zwischen den vorhergesagten

class PaperclipTrainer:
    '''Trainer Klasse zum Trainieren des Paperclip-Detektors'''
    def __init__(self, model, data_handler, learning_rate, num_epochs):
        self.model = model  # Das Modell, das trainiert werden soll
        self.data_handler = data_handler
        self.criterion_class = nn.CrossEntropyLoss()    # Klassifikationsverlust
        self.criterion_bbox = nn.MSELoss()            # Bounding-Box-Regression-Verlust
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.num_epochs = num_epochs

    def train(self):
        '''
        Trainiert das Modell mit den bereitgestellten Trainingsdaten und validiert es mit den Validierungsdaten.

        Diese Methode führt den Trainingsprozess über eine festgelegte Anzahl von Epochen durch. In jeder Epoche wird
        das Modell mit dem Trainingsdatensatz trainiert und anschließend mit dem Validierungsdatensatz getestet.
        Der gesamte Verlust (Klassifikations- und Bounding Box Verlust) wird berechnet und zur Optimierung der Modellparameter verwendet.
        Nach jeder Epoche werden die Trainings- und Validierungsverluste ausgegeben.

        Schritte:
        1. Initialisiert die Trainings- und Validierungsdataloader.
        2. Iteriert über die festgelegte Anzahl von Epochen.
        3. Für jede Epoche:
           a. Setzt das Modell in den Trainingsmodus.
           b. Initialisiert den kumulativen Verlust für die Epoche.
           c. Iteriert über jeden Batch im Trainingsdatensatz:
              - Verschiebt die Bilder und Zielwerte auf das verwendete Gerät (CPU oder GPU).
              - Setzt die Gradienten des Optimierers zurück.
              - Lässt die Bilder durch das Modell laufen, um Vorhersagen zu erhalten.
              - Berechnet den Klassifikationsverlust und den Bounding Box Verlust.
              - Summiert die Verluste und berechnet die Gradienten.
              - Aktualisiert die Modellparameter basierend auf den Gradienten.
              - Summiert den Verlust für die Epoche.
           d. Gibt den durchschnittlichen Verlust für die Epoche aus.
           e. Setzt das Modell in den Evaluierungsmodus.
           f. Initialisiert den kumulativen Validierungsverlust.
           g. Iteriert über jeden Batch im Validierungsdatensatz (ohne Gradientenberechnung):
              - Verschiebt die Bilder und Zielwerte auf das verwendete Gerät.
              - Lässt die Bilder durch das Modell laufen, um Vorhersagen zu erhalten.
              - Berechnet den Klassifikationsverlust und den Bounding Box Verlust.
              - Summiert die Verluste.
           h. Gibt den durchschnittlichen Validierungsverlust für die Epoche aus.
        '''
        train_loader, val_loader = self.data_handler.create_dataloaders()
        # Erstellt die Trainings- und Validierungsdataloader

        for epoch in range(self.num_epochs):
            self.model.train()  # Setzt das Modell in den Trainingsmodus
            running_loss = 0.0
            for images, targets in train_loader:    # Iteriert über jeden Batch im Trainingsdatensatz
                images = images.to(device)          # Verschiebt die Bilder auf das verwendete Gerät (CPU oder GPU)
                labels = targets['labels'].to(device)
                boxes = targets['boxes'].to(device).view(-1, 4)  # Targets in die richtige Form bringen

                self.optimizer.zero_grad()  # Setzt die Gradienten des Optimierers zurück
                classes, preds = self.model(images)

                # Predictions und Labels in die richtige Form bringen
                labels = labels.view(-1)
                classes = classes.view(-1, classes.shape[-1])

                loss_class = self.criterion_class(classes, labels)
                loss_bbox = self.criterion_bbox(preds, boxes)
                loss = loss_class + loss_bbox

                loss.backward()         # Berechnet die Gradienten
                self.optimizer.step()   # Optimiert die Modellparameter basierend auf den Gradienten
                running_loss += loss.item()

            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

            # In der Validierungsschleife wird das Modell auf dem Validierungsdatensatz getestet
            # Die Validierungsverluste fallen in der Regel höher aus als die Trainingsverluste, weil das Modell
            # auf Daten getestet wird, die es noch nicht gesehen hat.
            self.model.eval()   # Setzt das Modell in den Evaluierungsmodus
            val_loss = 0.0
            with torch.no_grad():   # Deaktiviert die Gradientenberechnung
                for images, targets in val_loader:
                    images = images.to(device)  # Verschiebt die Bilder auf das verwendete Gerät
                    labels = targets['labels'].to(device)   # Verschiebt die Labels auf das verwendete Gerät
                    boxes = targets['boxes'].to(device).view(-1, 4)  # Targets in die richtige Form bringen

                    classes, preds = self.model(images) # Lässt die Bilder durch das Modell laufen, um Vorhersagen zu erhalten

                    labels = labels.view(-1)    # Predictions und Labels in die richtige Form bringen
                    classes = classes.view(-1, classes.shape[-1])

                    loss_class = self.criterion_class(classes, labels)  # Berechnet den Klassifikationsverlust
                    loss_bbox = self.criterion_bbox(preds, boxes)    # Berechnet den Bounding Box Verlust
                    loss = loss_class + loss_bbox
                    val_loss += loss.item() # Summiert die Verluste

            print(f"Epoch {epoch + 1}, Validation Loss: {val_loss / len(val_loader)}")
        try:
            torch.save(self.model.state_dict(), r'A:\Paperclip Models\single_paperclips\CNN.pth')
        except Exception as e:
            print(f"Error saving model: {e}.")

# ----------------------------------------------------------------------------------------------------------------------
# Hier wird die Vorhersage des Modells implementiert. Die Funktion get_prediction nimmt ein Bild, das Modell und die
# Transformationen als Eingabe und gibt die vorhergesagten Bounding Boxes zurück.
# Die Funktion plot_image_with_boxes nimmt ein Bild, die Bounding Boxes und die Farbe der Büroklammer als Eingabe und
# zeigt das Bild mit den Bounding Boxes an.

# Die Funktion predict_and_save nimmt ein Bild, das Modell, die Transformationen, den Ausgabepfad und den Schwellenwert
# als Eingabe und gibt die gefilterten Bounding Boxes zurück.
def get_prediction(img_path, model, transform, threshold=0.5):
    '''
    Die Funktion predict_and_save nimmt ein Bild, das Modell, die Transformationen, den Ausgabepfad und den Schwellenwert
    als Eingabe und gibt die gefilterten Bounding Boxes zurück.
    '''

    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        classes, boxes = model(img)
    boxes = boxes.cpu().numpy()
    classes = classes.cpu().numpy()

    filtered_boxes = boxes[classes[:, 1] >= threshold]
    return filtered_boxes

# Die Funktion plot_image_with_boxes nimmt ein Bild, die Bounding Boxes und die Farbe der Büroklammer als Eingabe und
# zeigt das Bild mit den Bounding Boxes an.
def plot_image_with_boxes(img_path, boxes, color, output_path=None):
    img = Image.open(img_path).convert("RGB")
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    # Die Farben die von canny_hsv_classifier zurückgegeben werden in die Farben umgewandelt, die von matplotlib
    # verwendet werden, um die Bounding Boxes in der entsprechenden Farbe anzuzeigen.
    color_map = {
        'black': 'k',
        'white': 'w',
        'blue': 'b',
        'green': 'g',
        'yellow': 'y',
        'red': 'r',
        'undefined': 'm'
    }
    box_color = color_map.get(color, 'm')  # Default auf magenta, falls die Farbe nicht erkannt wird
    for box in boxes:
        rect = patches.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            linewidth=2, edgecolor=box_color, facecolor='none'
        )
        ax.add_patch(rect)
    if output_path:
        try:
            plt.savefig(output_path)
        except Exception as e:
            print(f"Error saving Image: {e}.")
    plt.show()

# Diese Funktion ist der Einstiegspunkt für das Farberkennungsmodul. Von hier aus werden die Funktionen zur
# Farberkennung und zur Vorhersage der Bounding Boxes aufgerufen.
def predict_and_save(img_path, model, transform, output_path, threshold=0.5):
    boxes = get_prediction(img_path, model, transform, threshold)
    detected_color = color_detection.canny_hsv_classifier(img_path)
    plot_image_with_boxes(img_path, boxes, detected_color, output_path)
# ----------------------------------------------------------------------------------------------------------------------
#Main script
if __name__ == '__main__':
    # Erzeugen einer transform Methode, um die Bilder in ein für das Modell passendes Format zu bringen
    transform = transforms.Compose([
        transforms.Resize((128, 128)), # Ändern der Auflösung auf 128x128
        # Umwandeln der Bilder in ein für Pytorch passendes Format:
        # Normalisierung der Pixelwerte auf den Bereich [0, 1]
        # Dimensionsänderung von (Höhe, Breite, Kanäle) zu (Kanäle, Höhe, Breite)
        transforms.ToTensor(),
        # Normalisierung der Pixelwerte auf den Bereich [-1, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Pfade zu Trainingsdaten
    img_dir = r'A:\Paperclip Models\single_paperclips\train3\train'
    annotations_dir = img_dir

    data_handler = DataHandler(img_dir, annotations_dir, val_ratio=0.2, batch_size=16, transform=transform)

    # CNN Modell initialisieren und auf spezifisches Gerät laden (GPU oder CPU)
    model = EfficientNetObjectDetector().to(device)

    trainer = PaperclipTrainer(model, data_handler, learning_rate=0.0005, num_epochs=40)

    # Abfrage, ob ein vortrainiertes Modell geladen werden soll und wenn ja, ob trainiert oder vorhergesagt werden soll
    while True:
        question_load_pretrained_model = input("Do you want to load a pre-trained model? (yes/no)\n")
        if question_load_pretrained_model == "yes":
            try:
                model.load_state_dict(torch.load(r'A:\Paperclip Models\single_paperclips\CNN.pth'))
                while True:
                    question_test_perdict = input("Pre-trained model loaded successfully.\n"
                                                  "Do you want to continue training or predict on "
                                                  "test images? (train/predict)\n")
                    if question_test_perdict == "train":
                        should_train = True
                        break
                    elif question_test_perdict == "predict":
                        should_train = False
                        break
                    else:
                        print("Invalid input. Please enter 'train' or 'predict'.\n")
                break
            except FileNotFoundError:
                print("No pre-trained model found.\n")
        elif question_load_pretrained_model == "no":
            should_train = True
            break
        else:
            print("Invalid input. Please enter 'yes' or 'no'.\n")

    if should_train:
        # starten des Trainings
        trainer.train()
    else:
        # Modell in den Auswertungsmodus versetzen
        model.eval()

    # Liste von Testbildern
    test_folder = r"A:\Paperclip Models\single_paperclips\predictions"
    test_img_paths = []
    for f in os.listdir(test_folder):
        if f.endswith('.jpg'):
            test_img_paths.append(os.path.join(test_folder, f))

    output_folder = os.path.join(test_folder, 'plotted_images')
    for img_path in test_img_paths:
        # ändern des Dateinamens, um das Ergebnisbild zu speichern
        output_file_name = (os.path.basename(img_path).replace('.jpg', '_result.jpg'))
        output_path = os.path.join(output_folder, output_file_name)
        predict_and_save(img_path, model, transform, output_path)

