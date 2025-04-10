# Lock-In - A Secure Attendance via Gaze and Blink Detection

A comprehensive AI-powered attendance system that verifies student presence based on gaze direction and blink detection using real-time video input. This intelligent solution minimizes proxy attendance and ensures accurate participation tracking through deep learning, computer vision, and an elegant Flask-based web interface.

---

## 📌 Table of Contents
- 🎯 Project Objective  
- 🧠 AI Backend Overview  
- 🔄 System Workflow  
- 📊 Dataset & Preprocessing  
- 📈 Model Performance  
- 🧰 Tech Stack  
- 🚀 Installation & Usage  
- 🖼️ Screenshots  
- 📦 Features  
- 🛠️ Future Enhancements  
- 📜 License & Contact  

---

## 🎯 Project Objective

This project aims to modernize classroom and remote learning attendance systems by implementing a non-intrusive, AI-based attendance verification system. It ensures that students are:

- Physically present (face detected)  
- Paying attention (gaze detection)  
- Awake and alert (blink detection)  

By leveraging gaze and blink behavior, the system eliminates proxy attendance and provides real-time attendance logs without manual intervention.

---

## 🧠 AI Backend Overview

### 🧪 Datasets Used

| Dataset             | Source | Classes             | Images per Class |
|---------------------|--------|----------------------|------------------|
| Drowsiness Detection| Kaggle | Open Eyes, Closed Eyes | 1000 each       |
| Gaze Detection      | Kaggle | Looking, Not Looking | 900 each         |

- **Eye region (64×64 grayscale)**: used to train an SVM classifier for blink detection.  
- **Face region (224×224 RGB)**: used to fine-tune MobileNetV2 for gaze classification.  

### 🧼 Preprocessing Techniques
- **Image Normalization**: Rescale pixel values to [0, 1]  
- **Data Augmentation**: Rotation, flipping, brightness  
- **Region Extraction**: Haar cascades/dlib to crop face & eyes  

---

## 🔄 System Workflow

### Step-by-Step Flow:
1. **Camera Input**  
   Live video feed captured through webcam.  
2. **Face Detection**  
   Uses OpenCV or Dlib to detect face bounding boxes.  
3. **Eye Region Extraction**  
   From detected faces, eyes are cropped out for blink and gaze analysis.  
4. **Blink Detection (SVM Classifier)**  
   Grayscale eye patches (64×64) passed to the trained SVM model.  
5. **Gaze Detection (MobileNetV2)**  
   RGB face regions (224×224) fed into the fine-tuned deep model.  
6. **Attendance Decision Logic**  
   If both eyes are open AND gaze is on screen for ≥10 seconds, attendance is logged.  
7. **Data Logging & Reporting**  
   Saves attendance data into CSV or database, optionally generating reports.  

---

## 📊 Dataset & Preprocessing

Data preparation is key to model performance. Here’s what we did:

- Eye crops from face images were resized to 64×64 for blink classification.  
- Face region resized to 224×224 for MobileNetV2 input.  
- Normalized all images.  
- Applied data augmentation to increase robustness.  

### 📸 Region of Focus
- **Eye region → Blink model**  
- **Upper face region → Gaze model**

---

## 📈 Model Performance

### Blink Detection (SVM)
- ✅ Accuracy: 98.5%  
- 🎯 Near-perfect detection of open/closed eyes.  

### Gaze Detection (MobileNetV2)

| Model        | Accuracy | Inference Time | Memory Use |
|--------------|----------|----------------|------------|
| MobileNetV2  | 85%      | 0.117ms/frame  | 8.9 MiB    |
| ResNet50     | 82.7%    | 0.223ms        | 97.2 MiB   |
| DenseNet169  | 83.1%    | 0.307ms        | 124.7 MiB  |

MobileNetV2 chosen for its lightweight and fast inference, ideal for real-time processing on edge devices.

---

## 🧰 Tech Stack

| Layer      | Technology Used                        |
|------------|-----------------------------------------|
| Frontend   | HTML5, CSS3, Bootstrap 5               |
| Backend    | Flask (Python)                         |
| AI Models  | scikit-learn (SVM), TensorFlow/Keras   |
| CV Tools   | OpenCV, Dlib                           |
| Data Viz   | Matplotlib, Seaborn                    |
| Deployment | Localhost/Flask Web App                |

---

🖼️ Screenshots
Add screenshots in the screenshots/ folder and update here.

🎥 Live Detection Interface

![s1](https://github.com/user-attachments/assets/c132a4fb-9eee-4528-a1ed-fec1c3e9ccaf)

📊 Attendance Log View
![s2](https://github.com/user-attachments/assets/c69b2115-6c41-4564-b859-63daf154f2df)

📦 Features

✅ Real-time gaze and blink detection

🎯 AI model trained on curated datasets

🔐 Proxy attendance prevention

⏱️ Timer-based logic for reliable presence detection

📁 CSV logging of attendance

🧪 Easy integration with future face recognition

💡 Flask-based modular architecture


🛠️ Future Enhancements

🔒 Integrate face recognition (FaceNet / VGG-Face)

🌍 Deploy on Raspberry Pi or edge devices

☁️ Add cloud-based dashboards for analytics

👨‍🏫 Connect with LMS systems for automated reporting

📊 Add head pose and emotion detection for better engagement metrics


📜 License & Contact

This project is licensed under the MIT License.

Created by S.R.Rosheni
📧 rosheniramesh@gmail.com



