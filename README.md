# 🧬 AI-Powered Virtual Screening Interface

This project provides a user-friendly web application built with Streamlit for performing AI-driven virtual screening of chemical compounds. Users can upload a list of molecules in SMILES format, and the application will predict their biological activity using a pre-trained machine learning model. The results are presented in an interactive dashboard and can be downloaded as a comprehensive PDF report and a CSV file.

The application is designed to be model-agnostic, supporting models saved with Scikit-learn (`.joblib`, `.pkl`) and Keras/TensorFlow (`.h5`).

 <!-- 👈 **مهم:** استبدل هذا الرابط بلقطة شاشة لتطبيقك! -->

---

## 🚀 Features

-   **Easy File Upload:** Upload a simple CSV file with a `smiles` column.
-   **High-Throughput Prediction:** Rapidly predicts the activity for thousands of compounds.
-   **Interactive Dashboard:** View, sort, and analyze the top-ranked compounds.
-   **ADME/T & Physicochemical Properties:** Automatically calculates key properties like Molecular Weight (MW), LogP, HBD/HBA, and checks for Rule-of-Five (RO5) violations.
-   **Rich Visualizations:** Generates multiple plots to analyze the results, including:
    -   Predicted Activity Bar Chart
    -   Chemical Space (MW vs. LogP)
    -   Property Distribution Histograms
    -   t-SNE for visualizing chemical diversity
    -   Correlation Heatmaps
-   **Comprehensive PDF Reports:** Generates a professional, multi-page PDF report with summary statistics, all visualizations, and detailed pages for each top compound.
-   **Data Export:** Download the ranked list of compounds as a clean CSV file.
-   **Model Agnostic:** Easily swap between different trained models (`.joblib`, `.h5`).

---

## 🛠️ Technology Stack

-   **Backend & ML:** Python, Scikit-learn, TensorFlow/Keras, RDKit, Pandas, NumPy
-   **Frontend:** Streamlit
-   **Reporting:** FPDF2

---

## 📋 Prerequisites

-   Python 3.8+
-   Conda or a virtual environment manager (recommended)

---

## ⚙️ Installation & Setup

Follow these steps to set up and run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
Use code with caution.
Markdown
2. Set Up the Environment
It is highly recommended to use a Conda environment to manage dependencies, especially for RDKit and TensorFlow.
# Create a new conda environment
conda create -n v-screening python=3.9 -y

# Activate the environment
conda activate v-screening

# Install dependencies using pip
pip install -r requirements.txt
Use code with caution.
Bash
Note: A requirements.txt file is needed. You can generate one using pip freeze > requirements.txt. Here is a sample requirements.txt based on your app.py:
streamlit
pandas
numpy
scikit-learn
joblib
tensorflow
fpdf2
matplotlib
seaborn
rdkit
pillow
Use code with caution.
3. Place Your Trained Model
The application expects the trained model and feature selector to be in a specific directory:
Create a folder named saved_model in the root directory of the project.
Place your trained model file inside this folder. It should be named best_model.joblib, best_model.pkl, or best_model.h5.
Place your feature selector file (from Scikit-learn's SelectFromModel) inside the same folder. It must be named feature_selector.pkl.
The final structure should look like this:
your-repository-name/
├── saved_model/
│   ├── best_model.joblib   # (or .h5, .pkl)
│   └── feature_selector.pkl
├── app.py
├── README.md
└── requirements.txt
Use code with caution.
▶️ Running the Application
Once the environment is set up and the model is in place, you can run the Streamlit app with the following command:
streamlit run app.py
Use code with caution.
Bash
The application will open in your default web browser.
USAGE
Open the App: Navigate to the local URL provided by Streamlit (usually http://localhost:8501).
Upload Data: In the sidebar, click "Browse files" to upload a CSV file. The file must contain a header and a column named smiles.
Configure Options:
Use the slider or the number input to select how many of the top-ranked compounds you want to include in the report (e.g., Top 10, Top 50).
Start Screening: Click the "🚀 Start Prediction & Screening" button. The app will process the compounds, make predictions, and generate results.
Analyze Results:
An interactive table showing the top compounds and their properties will appear.
Use the download buttons to get the results as a CSV file or the full analysis as a PDF report.
Preview the generated charts in the "Chart Preview for Report" section at the bottom of the page.
📁 Project Structure
.
├── app.py                      # Main Streamlit application script
├── saved_model/                # Directory for storing the trained model and selector
│   ├── best_model.h5           # Example: Keras model
│   └── feature_selector.pkl    # The feature selector object
├── requirements.txt            # Python dependencies
└── README.md                   # This file
Use code with caution.
🧠 Model & Feature Engineering Details
Fingerprints: The application uses Morgan Fingerprints (a circular fingerprint similar to ECFP) with a radius of 3 and 2048 bits. These parameters are hard-coded and must match the parameters used during model training.
Feature Selection: A pre-trained SelectFromModel object (feature_selector.pkl) is used to reduce the dimensionality of the fingerprint vectors before feeding them to the prediction model. This is crucial for performance and must also match the training pipeline.
🤝 Contributing
Contributions are welcome! If you have suggestions for improvements or find any issues, please feel free to open an issue or submit a pull request.
