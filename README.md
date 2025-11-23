AgriGuard 🐛🌾

CNN-Based Pest Detection and Advisory System

AgriGuard is a computer vision–based web application that helps farmers and agricultural stakeholders identify crop pests from images and receive basic guidance on how to manage them. It combines a Convolutional Neural Network (CNN) trained on the IP102 dataset with a user-friendly web interface and an intelligent recommendation module that can summarise pest management options in simple language.

This project was developed as a Final Year Project for the Bachelor of Informatics and Computer Science at Strathmore University.

📚 Table of Contents

Project Overview

Problem Statement

Objectives

Key Features

System Architecture

Tech Stack

Dataset: IP102

Data Preprocessing & Augmentation

Handling Class Imbalance

Model Design & Training

Model Performance & Evaluation Metrics

Repository Structure

Installation & Setup

Running the Application

Using AgriGuard

API Overview

Security, Privacy & Ethical Considerations

Limitations & Future Work

Project Mapping to Academic Chapters

Citing the Dataset & Related Work

License

Acknowledgements

🌍 Project Overview

Pest infestations are a major threat to global food security. Farmers, especially smallholder farmers, often struggle to:

Identify the exact pest species affecting their crops

Decide when and how to intervene

Access timely, expert advice from agronomists or extension officers

With the growth of smartphones and affordable internet, it is now possible to support farmers using image-based pest detection systems. By capturing a photo of a pest on a leaf, stem, or fruit, a farmer can receive a preliminary identification and simple guidance in seconds.

AgriGuard is a web-based decision-support system that:

Accepts pest images from smartphones, tablets, or computers

Uses a trained Convolutional Neural Network (CNN) based on EfficientNetV2 to classify the pest

Displays the predicted pest class together with a confidence score

Provides a short description of the pest and management recommendations drawn from a pest knowledge base and an AI-powered recommendation component

AgriGuard focuses on pest detection only (not weed detection, not diseases) and is designed to be easily accessible via a standard web browser.

❗ Problem Statement

In an ideal agricultural setting, farmers, regardless of scale or location, should have timely and accurate information about pests affecting their crops so they can respond appropriately and protect their yields. With the growing accessibility of smartphones and digital tools, technology can support this need through reliable image-based detection models that enable users to make quick and informed decisions.

Although neural network–based models and smartphone-enabled pest detection tools have been explored in research, pest infestations continue to cause serious crop losses globally. Many farmers still face difficulties in identifying pest species promptly due to limited access to expert knowledge and diagnostic tools. Diverse pest types, changing climatic conditions, and inadequate agricultural advisory services further complicate early detection and management.

This gap leads to delayed identification, excessive or inappropriate pesticide use, and reduced agricultural productivity. AgriGuard addresses this challenge by providing a CNN-based web system that can identify pests directly from uploaded images and present basic, actionable advisory information.

🎯 Objectives
Main Objective

To develop a Convolutional Neural Network (CNN)-based image classification model that accurately identifies common crop pests directly from uploaded pest images and provides corresponding pest management recommendations. 

Specific Objectives

i.	To investigate pest distribution patterns and identification approaches.
ii.	To analyse existing deep learning-based models used for pest identification. 
iii.	To design and train a Convolutional Neural Network (CNN) model capable of identifying pests from uploaded pest images and providing context-appropriate remedy recommendations.
iv.	To test and validate the model through a web platform and evaluate its performance based on metrics like accuracy, precision, and recall.


✨ Key Features
1. Pest Image Upload

Upload from:

Local gallery / file system

Supported image formats:

.jpg, .jpeg, .png 

2. CNN-Based Pest Identification

Uses a pre-trained EfficientNetV2 model fine-tuned on the IP102 dataset.

Outputs:

Predicted pest species name

Internal class ID

Confidence score (probability)

Optionally, the top-3 most likely classes

3. Advisory & Recommendation Module

For each predicted pest, the system displays:

A short textual description

Common host crops

Notable damage symptoms

High-level management suggestions, with emphasis on:

Integrated Pest Management (IPM)

Non-chemical options where possible

A recommendation component powered by the GPT API (e.g., OpenAI) can:

Rephrase technical remedial measures into farmer-friendly language

Provide structured advice sections such as “Monitoring”, “Cultural Control”, and “Chemical Control (if necessary)”

Disclaimer: AgriGuard is a decision-support and educational tool. It does not replace local agricultural regulations, pesticide labels, or professional advice from agronomists and extension officers.

4. User-Friendly Web Interface

Responsive layout that works on:

Mobile phones

Tablets

Desktops

Simple workflow:

Upload pest image

Click Predict

View pest name, confidence, and recommended actions

5. (Optional) Administrative Features

Dashboard view (optional), including:

Summary of prediction counts

Ability to update textual pest descriptions and management notes

Model file management: replace the trained model without changing frontend code.

🏗 System Architecture

AgriGuard adopts a modular, service-oriented architecture with the following layers:

Presentation Layer (Frontend)

Built using Next.js (React) and Tailwind CSS.

Handles user interaction, image selection, and display of prediction results.

Communicates with the backend via RESTful HTTP requests.

Application Layer (Backend API)

Implemented using Python 3 and FastAPI.

Exposes endpoints for:

GET /health – health check

POST /predict – pest image classification

Performs request validation, logging, and simple error handling.

Machine Learning / Inference Layer

Contains logic for:

Image preprocessing (resize, normalization)

Model loading and warm-up

Forward pass through the EfficientNetV2 classifier

Returns the predicted class index and probability distribution.

Recommendation & Knowledge Layer

Pest knowledge base stored as JSON/CSV or in a database table, including:

Pest name

Description

Host crops

Management guidance

GPT-based recommendation service that:

Accepts the pest label and basic facts as input

Generates clear, concise farmer-oriented advisory text.

Data & Storage Layer

Stores:

Trained model file (e.g., agriguard_efficientnetv2.h5 or .pt)

Temporary or persistent uploaded images (optional)

Logs and configuration files

Database (if configured) for:

Users/admins

Prediction history

Pest content.

Infrastructure & DevOps (Optional)

Can be deployed on:

An Ubuntu server/VM (e.g., on Azure)

Shared hosting with a separate API service

Uses:

Uvicorn / Gunicorn for production API serving

Nginx or a reverse proxy

Basic CI/CD pipeline to build and deploy backend and frontend.

🧰 Tech Stack
Frontend

Framework: Next.js (React)

Language: TypeScript or JavaScript

Styling: Tailwind CSS

HTTP Client: Axios or Fetch API

Build Tool: Next.js built-in tooling (next build)

Backend

Language: Python 3

Framework: FastAPI

Server: Uvicorn (development) / Uvicorn/Gunicorn (production)

Environment: venv virtual environment

Machine Learning

Framework: TensorFlow / Keras (with EfficientNetV2)

Core libraries: NumPy, Pandas, scikit-learn

Image handling: Pillow (PIL) and/or OpenCV

Data augmentation: Keras preprocessing layers or Albumentations

Other

Version Control: Git & GitHub

Deployment: Ubuntu VM / cloud hosting, or local server

Recommendation API: GPT model via an API (e.g., OpenAI)

Documentation and Diagrams: PlantUML, draw.io, and Markdown

🐞 Dataset: IP102

The model is trained using a dataset derived from IP102, a widely used large-scale benchmark dataset for insect pest recognition.

Name: IP102 Dataset

Mirror used: IP02/IP102 mirror on Kaggle

Source platform: Kaggle

Access link:
https://www.kaggle.com/datasets/rtlmhjbn/ip02-dataset

Dataset Characteristics

Number of classes: 102 agricultural pest species

Number of images: Over 75,000 images in the full benchmark

Image properties:

Real field conditions

High variability in:

Illumination

Backgrounds

Pest pose, orientation, and scale

Different crop hosts and growth stages

Data Split

For this project, the dataset is split into:

Training set: Approximately 70% of the data

Validation set: Approximately 15% of the data

Test set: Approximately 15% of the data

The exact split is implemented in the training notebook (e.g., using stratified splitting to preserve class proportions where possible).

🧹 Data Preprocessing & Augmentation

Before training the CNN, the dataset goes through several preprocessing steps:

1. Data Cleaning

Removal or exclusion of:

Corrupted image files

Extremely low-resolution images that hinder recognition

Duplicate or near-duplicate images, where detected

Normalisation of directory and file naming conventions, ensuring that each class has its own folder and label.

2. Label Encoding

Mapping from class names (e.g., "H. armigera") to integer indices (e.g., 0–101).

Creation of a dictionary such as:

class_index_to_name = {
  0: "Helicoverpa armigera",
  1: "Spodoptera litura",
  ...
}


Labels are one-hot encoded or kept as integer labels depending on the chosen loss function.

3. Image Preprocessing

Resize all images to a fixed resolution compatible with EfficientNetV2, e.g. 224 × 224 pixels.

Color mode: Convert to RGB (3 channels).

Scaling/Normalization:

Rescale pixel values from [0, 255] to [0, 1].

Optionally apply ImageNet-style normalization if using a pretrained backbone (mean/std normalization).

4. Data Augmentation

To improve generalisation and increase robustness to variations in real-world conditions:

Random horizontal flips

Small random rotations (e.g., ±15°)

Random zooming and cropping

Random brightness and contrast adjustments

Random shifts and slight perspective changes

Augmentations are applied only on the training set; the validation and test sets are kept untouched to provide realistic evaluation.

⚖ Handling Class Imbalance

The IP102 dataset is highly imbalanced: some pest classes have many more images than others. To address this:

Class Distribution Analysis

Class frequencies (images per class) are computed and visualised using bar charts or histograms.

Extremely rare classes are identified.

Sampling Strategy

Oversampling of minority classes is performed in the training set so that:

Classes with very few images are duplicated more often in the training batches.

Alternatively, class weights can be calculated and passed to the training loop so that:

Misclassifying minority-class images incurs a higher loss penalty.

Implementation

During training, a sampler or a class-weight dictionary is used so that the model “sees” a more balanced representation of classes over the epochs, reducing bias towards majority classes.

This approach improves recall for underrepresented pests while preserving performance on frequent ones.

🧠 Model Design & Training
Model Architecture

The AgriGuard model is based on EfficientNetV2, a family of CNN architectures known for achieving high accuracy with relatively few parameters.

Base model:

Pretrained EfficientNetV2 variant (e.g., EfficientNetV2-S) with ImageNet weights.

The top (original classification head) is removed.

Custom classification head:

Global Average Pooling layer

One or more fully connected (Dense) layers with ReLU activation

Dropout layers to reduce overfitting

Final Dense layer with:

Number of units = 102 (one per pest class)

Softmax activation for multi-class probability output

Training Configuration (Typical)

Loss function: Categorical Cross-Entropy or Sparse Categorical Cross-Entropy

Optimizer: Adam (adaptive learning rate)

Learning rate: Small initial value (e.g., in the order of 1e-4) with possible scheduling

Batch size: Typically between 16 and 32 images per batch

Number of epochs: Trained for multiple epochs with:

Early stopping based on validation loss to avoid overfitting

Model checkpointing to save the best weights

Training Pipeline

Load and split the IP102 dataset into training, validation, and test sets.

Define data generators or tf.data pipelines with preprocessing and augmentation.

Build the EfficientNetV2-based model and attach the custom classification head.

Compile the model with selected optimizer, loss, and metrics.

Train the model:

Monitor training/validation loss and accuracy.

Apply early stopping and learning rate scheduling as needed.

Evaluate the final saved model on the test set and generate:

Confusion matrix

Classification report (precision, recall, F1-score)

The training process is captured and documented in a Jupyter Notebook (e.g., notebooks/agriguard_training.ipynb), including plots of accuracy and loss over epochs.

📊 Model Performance & Evaluation Metrics

The model is evaluated on a held-out test set, with the following metrics:

Accuracy (Top-1):

Proportion of test images whose most probable predicted class matches the true class.

Top-3 Accuracy (if reported):

Proportion of test images where the true class appears in the top three predicted classes.

Precision (per class and macro-averaged):

Out of all images predicted as a particular pest, how many are correct?

Recall (per class and macro-averaged):

Out of all images belonging to a particular pest, how many are correctly identified?

F1-Score:

Harmonic mean of precision and recall, useful for imbalanced datasets.

Confusion Matrix:

A matrix visualising how often each pest is confused with other pests.

Highlights visually similar species that the model struggles to distinguish.

In qualitative terms, the trained EfficientNetV2 pest classifier demonstrates:

Strong performance on well-represented, visually distinctive pests.

Moderate challenges on very rare classes or those that are visually similar to others (e.g., larvae with similar coloration).

Improved minority-class performance compared to an unbalanced baseline due to the implemented sampling strategy.

Exact numerical values can be reported in the accompanying academic report, referencing the confusion matrix and classification report generated in the training notebook.

📂 Repository Structure

A typical repository structure for AgriGuard is as follows:

agriguard/
├── backend/
│   ├── app/
│   │   ├── main.py                # FastAPI entrypoint
│   │   ├── api/
│   │   │   └── routes.py          # API route definitions (predict, health, etc.)
│   │   ├── core/
│   │   │   ├── config.py          # Settings, environment variables
│   │   │   └── utils.py           # Utility functions
│   │   ├── ml/
│   │   │   ├── model.py           # Model loading and inference
│   │   │   └── preprocessing.py   # Image preprocessing & augmentation
│   │   └── schemas/
│   │       └── responses.py       # Pydantic models for API responses
│   ├── models/
│   │   └── agriguard_efficientnetv2.h5   # Trained CNN model weights
│   ├── notebooks/
│   │   └── agriguard_training.ipynb      # Training and evaluation notebook
│   ├── requirements.txt
│   └── README_backend.md
├── frontend/
│   ├── src/ or pages/             # Next.js pages and app entrypoints
│   ├── components/                # Reusable UI components
│   ├── styles/                    # Global styles, Tailwind config
│   ├── public/                    # Static assets (logo, icons)
│   ├── package.json
│   └── README_frontend.md
├── docs/
│   ├── diagrams/                  # UML, ERD, system architecture images
│   ├── screenshots/               # UI screenshots, confusion matrix, etc.
│   └── report/                    # Project report files (optional)
├── .gitignore
└── README.md


The exact filenames and folder structure may vary slightly depending on your implementation, but this layout supports separation of concerns between backend, frontend, ML components, and documentation.

🔧 Installation & Setup
1. Prerequisites

Ensure the following are installed:

Python: 3.9 or later

Node.js: 18 or later

npm or yarn

Git

(Optional) A GPU with CUDA support for faster training (not strictly required for inference)

Clone the project:

git clone https://github.com/<username>/agriguard.git
cd agriguard


Replace <username> with your GitHub username or the relevant repository path.

2. Backend Setup (FastAPI)

From the project root:

cd backend


Create and activate a virtual environment.

On Windows (PowerShell):

python -m venv venv
venv\Scripts\activate


On Linux/macOS:

python -m venv venv
source venv/bin/activate


Install Python dependencies:

pip install --upgrade pip
pip install -r requirements.txt


Ensure the trained model file is placed in:

backend/models/agriguard_efficientnetv2.h5


(or the filename you configured in your settings).

Create a .env file in backend/ with environment variables such as:

MODEL_PATH=./models/agriguard_efficientnetv2.h5
ALLOWED_ORIGINS=http://localhost:3000
OPENAI_API_KEY=your-openai-api-key-here
PREDICTION_THRESHOLD=0.5
LOG_LEVEL=info

3. Frontend Setup (Next.js)

From the project root:

cd frontend


Install Node dependencies:

npm install
# or
yarn install


Create a .env.local file in frontend/ (for Next.js) with:

NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
NEXT_PUBLIC_APP_NAME=AgriGuard

▶ Running the Application
1. Start the Backend API

From backend/ with the virtual environment activated:

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload


The backend should now be running at:

http://localhost:8000

For FastAPI interactive documentation:

Swagger UI: http://localhost:8000/docs

ReDoc: http://localhost:8000/redoc

2. Start the Frontend

In another terminal, from frontend/:

npm run dev
# or
yarn dev


The frontend will start at:

http://localhost:3000

Open this URL in your browser to access the AgriGuard web interface.

📲 Using AgriGuard

Open the web application at http://localhost:3000 or the deployed URL.

Navigate to the Pest Detection page (usually the home page).

Click Upload Image or Take Photo:

On mobile, you can capture a fresh photo using the camera.

On desktop, choose an existing image file of the pest.

After selecting the image, click Predict.

The system sends the image to the backend and:

Preprocesses the image

Runs the CNN model for inference

Retrieves the pest name, confidence score, and recommendations

The results page displays:

Pest name (e.g., Helicoverpa armigera)

Confidence score (e.g., 0.87 or 87%)

A brief description of the pest

Recommended management options in simple language

If the confidence is low or the result is marked as “uncertain”, the user is advised to:

Retake the photo with better lighting/focus

Capture the pest from closer or different angles

Consult a professional agronomist or extension officer.

📡 API Overview
GET /health

Description:
Simple health check endpoint.

Request:

GET /health


Response (200 OK):

{
  "status": "ok",
  "message": "AgriGuard API is running"
}

POST /predict

Description:
Accepts an image file and returns the predicted pest class and advice.

Request:

Method: POST

URL: http://localhost:8000/predict

Body: multipart/form-data with field file containing the image.

Example using curl:

curl -X POST "http://localhost:8000/predict" \
  -F "file=@/path/to/pest_image.jpg"


Sample Response (200 OK):

{
  "pest_name": "Helicoverpa armigera",
  "class_id": 12,
  "confidence": 0.87,
  "top_k": [
    { "pest_name": "Helicoverpa armigera", "confidence": 0.87 },
    { "pest_name": "Spodoptera litura", "confidence": 0.07 },
    { "pest_name": "Mythimna separata", "confidence": 0.03 }
  ],
  "advice": {
    "summary": "Helicoverpa armigera is a polyphagous pest that attacks many crops including cotton, maize, and tomatoes.",
    "management": [
      "Regularly scout fields and use pheromone traps to monitor adult moth populations.",
      "Encourage natural enemies such as parasitoid wasps and predatory bugs.",
      "Only apply selective insecticides when necessary and in accordance with local agricultural extension guidelines."
    ]
  }
}


If the confidence score is below the configured PREDICTION_THRESHOLD, the API can return a response that indicates low confidence and suggests the user seek expert advice.

🔐 Security, Privacy & Ethical Considerations

Data Privacy:

Uploaded images are used solely for prediction and optional logging.

The system can be configured to:

Delete images immediately after inference, or

Store anonymised samples for improvement and research (with consent).

Model & API Security:

Backend endpoints should be protected behind a secure server (HTTPS) in production.

CORS rules limit which frontends can call the API.

API keys (e.g., for GPT) are stored as environment variables, never hard-coded in the repository.

Ethical Use:

Recommendations focus on Integrated Pest Management (IPM) principles.

The system avoids hard-coding specific pesticide brand names and encourages users to:

Follow national regulations

Consult local experts

The README and UI include disclaimers so users understand this is an advisory tool, not a substitute for professional agronomic services.

⚠ Limitations & Future Work
Limitations

Image Quality Sensitivity:

Performance may degrade on:

Very blurry, dark, or low-resolution images

Images where the pest is extremely small relative to the frame

Scope:

Focuses on pest identification only, not:

Crop diseases

Nutrient deficiencies

Weed recognition

Geographical Adaptation:

Model trained on IP102, which may not cover every local pest variant or species in all regions.

Some pests present in local contexts might not be in the dataset.

Advisory Depth:

Recommendations are high-level and generic.

They may not perfectly account for:

Specific crop varieties

Regional pest resistance patterns

Local pesticide registration and regulation.

Future Work

Extend the model to include:

Plant disease detection and nutrient deficiency diagnosis.

Develop a mobile app (Android/iOS) that uses the same backend.

Integrate localized advisory content for specific regions and crops.

Continuously improve the model using:

Feedback from real users

Additional labelled pest images from local farms

Explore more advanced architectures:

Vision Transformers (ViT)

Ensemble models combining CNNs and transformers.

Add multi-language support for different regions and literacy levels.

🧾 Project Mapping to Academic Chapters

This project can be mapped to a standard academic report structure as follows:

Chapter 1: Introduction

Background of pest management and digital agriculture

Problem statement

Objectives and research questions

Justification and scope of the study

Chapter 2: Literature Review

Overview of pest detection methods (traditional and digital)

Machine learning and deep learning in agriculture

Review of CNN architectures (including EfficientNetV2)

Summary of existing pest detection systems and their limitations

Summary of the IP102 dataset and related work

Chapter 3: System Analysis & Design

Requirements analysis (functional and non-functional)

Use case diagrams and descriptions

Sequence diagrams

Class diagram

Entity-Relationship Diagram (ERD)

System architecture diagram

Wireframes of key user interfaces

Chapter 4: Methodology

Dataset description and acquisition (IP102)

Data preprocessing, augmentation, and handling class imbalance

Model design, training strategy, and hyperparameters

Tools and technologies used (FastAPI, Next.js, TensorFlow/Keras)

Evaluation metrics and experimental setup

Chapter 5: System Implementation & Testing

Implementation of the backend API and ML inference logic

Implementation of the frontend (user interface)

Integration of recommendation module (GPT API)

Testing strategy:

Unit tests (if implemented)

Integration tests

Model evaluation results (confusion matrix, metrics)

Chapter 6: Conclusion & Recommendations

Summary of findings

Contributions of AgriGuard

Limitations encountered

Recommendations for future work and deployment in real farming contexts

Appendices

Complete code listings (where necessary)

Additional diagrams or screenshots

Ethical clearance and consent forms (if applicable)

📚 Citing the Dataset & Related Work

If you use or publish work based on this project, you should cite the IP102 dataset and relevant literature.

IP102 Dataset:

@inproceedings{wu2019ip102,
  title     = {IP102: A Large-Scale Benchmark Dataset for Insect Pest Recognition},
  author    = {Wu, J. and others},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2019}
}


You may also cite:

Papers reviewing deep learning in agriculture

The EfficientNetV2 paper

Any other sources you used for pest management recommendations and design decisions.

📜 License

This project is licensed under the MIT License.

You are free to use, modify, and distribute the code, provided that the original copyright notice and this permission notice are included in all copies or substantial portions of the software.

A sample LICENSE file:

MIT License

Copyright (c) 2025 AgriGuard




(Include the full MIT License text in an actual LICENSE file in your repository.)

🙏 Acknowledgements

The creators of the IP102 dataset for making a large, high-quality pest dataset publicly available.

The open-source community behind TensorFlow, Keras, FastAPI, Next.js, Tailwind CSS, and many other libraries used in this project.

Strathmore University, for academic guidance and support.

The project supervisor, lecturers, and classmates who provided feedback on model design, system architecture, and evaluation.

Farmers and agricultural practitioners whose real-world challenges inspired the need for tools like AgriGuard.
