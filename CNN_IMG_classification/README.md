-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Paperclip Detection and Color Classification**

This repository contains a system for detecting and classifying paperclips in images. It combines a deep learning model based on EfficientNet-B0 for object detection with a 
traditional image processing approach for color classification.
The detection model identifies paperclips and predicts their bounding boxes using transfer learning in PyTorch. For each detected object, the color is classified using Canny 
edge detection and HSV color space analysis. Supported colors include red, green, blue, yellow, black, and white.

During prediction, bounding boxes are drawn around paperclips in their detected color. The system is modular, easy to extend, and was developed by Lennart Theiler and 
David Pfliehinger as part of a university project.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Project Structure**

- modell_erkennung_bueroklammern.py: Contains the full pipeline for training and testing a paperclip object detection model using PyTorch and EfficientNet.
- color_detection.py: A module used to classify the color of paperclips detected in images using Canny edge detection and HSV color mapping.

