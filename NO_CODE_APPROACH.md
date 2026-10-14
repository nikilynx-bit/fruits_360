# No-Code AI Project Lifecycle

The No-Code approach utilizes platforms like **Google Teachable Machine**, **Lobe.ai**, **Azure Custom Vision**, or **Clarifai** to train models without writing Python code.

## 1. Scoping Project
* **Objective:** Create a lightweight, mobile-friendly fruit classifier for a diet-tracking app.
* **Success Criteria:** >90% accuracy on unseen test images with an inference time under 100ms.
* **Constraints:** Model must be exportable to TensorFlow Lite (TFLite) or CoreML for mobile deployment.

## 2. Data Preparation
* **Organization:** Organize the downloaded Kaggle dataset into folders by class name (e.g., `Apple Braeburn/`, `Banana/`).
* **Sampling:** AutoML platforms usually have upload limits (e.g., Teachable Machine limits you to ~2,000 images). You must sample a representative subset of images from the Training and Test directories of Fruits-360.
* **Upload:** Drag and drop the folders into the web-based interface of your chosen AutoML platform.

## 3. Training (Modeling)
* Click "Train Model" in the UI. The platform handles the architecture selection (typically a MobileNetV2 variant), transfer learning, and hyperparameter tuning automatically.
* Monitor the real-time training loss and accuracy graphs provided by the platform.

## 4. Evaluation
* **Preview:** Use the built-in webcam or file upload feature to test the model on images from the Test set.
* **Metrics:** Review the platform's "Confusion Matrix" or "Precision/Recall" tabs to identify which fruits are being confused (e.g., mistaking a "Mandarin" for an "Orange").

## 5. Deployment
* Export the model in the required format (e.g., `.tflite` for Android, `.mlmodel` for iOS, or an API endpoint via Azure Custom Vision).
* Integrate the model into a front-end application (React Native, Flutter, or a web app).
