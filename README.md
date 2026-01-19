Acute Lymphoblastic Leukemia (ALL) Detector
This project is a simple web‑based tool that uses a Teachable Machine image classification model to help identify whether a blood smear image appears Healthy or shows signs of Acute Lymphoblastic Leukemia (ALL).

It includes two detection modes:
- Webcam Detection – Uses your camera to classify live images
- Image Upload Detection – Allows you to upload a blood smear image for analysis

The model outputs:
- The probability (%) for each class
- A final result showing the most likely classification

Important Disclaimer
This tool is NOT a medical diagnostic device.
It is for educational and experimental purposes only.
Always consult a licensed medical professional for real medical evaluation or diagnosis.

Model Training
The model was trained using Google Teachable Machine with two classes:
- ALL (Acute Lymphoblastic Leukemia)
- Healthy
After training, the model achieved ~75% accuracy on validation images.
