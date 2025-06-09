
# 🧬 AI-Powered Virtual Screening Interface

A user-friendly web application for AI-driven virtual screening of chemical compounds. Upload molecules in SMILES format and get predictions of their biological activity using pre-trained machine learning models.

![App Screenshot](https://via.placeholder.com/800x500?text=Virtual+Screening+App+Screenshot)  
*Replace with actual screenshot of your application*

---

## 🚀 Key Features

### Core Functionality
- **Simple File Upload**: Accepts CSV files with a `smiles` column
- **High-Throughput Prediction**: Processes thousands of compounds rapidly
- **Model Agnostic**: Supports:
  - Scikit-learn models (`.joblib`, `.pkl`)
  - Keras/TensorFlow models (`.h5`)

### Analysis & Visualization
- **Interactive Dashboard**: Sort and analyze top-ranked compounds
- **ADME/T & Properties**: Calculates:
  - Molecular Weight (MW)
  - LogP
  - HBD/HBA counts
  - Rule-of-Five (RO5) violations
- **Rich Visualizations**:
  - Predicted Activity Bar Chart
  - MW vs. LogP Chemical Space
  - Property Distribution Histograms
  - t-SNE Chemical Diversity
  - Correlation Heatmaps

### Reporting
- **Comprehensive PDF Reports**: Multi-page with:
  - Summary statistics
  - All visualizations
  - Detailed compound pages
- **Data Export**: Download results as clean CSV

---

## 🛠️ Technology Stack

| Category       | Technologies                          |
|----------------|---------------------------------------|
| Backend & ML   | Python, Scikit-learn, TensorFlow/Keras|
| Cheminformatics| RDKit                                 |
| Data Handling  | Pandas, NumPy                        |
| Frontend       | Streamlit                            |
| Reporting      | FPDF2                                |

---

## 📋 Prerequisites

- Python 3.8+
- Conda (recommended) or virtualenv

---

## ⚙️ Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Create Conda Environment
```bash
conda create -n v-screening python=3.9 -y
conda activate v-screening
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

*Sample `requirements.txt`:  
```
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
```

### 4. Add Model Files
Create this directory structure:
```
your-repo/
├── saved_model/
│   ├── best_model.joblib   # or .h5/.pkl
│   └── feature_selector.pkl
└── ...
```

---

## ▶️ Running the Application

```bash
streamlit run app.py
```
The app will open at `http://localhost:8501`

---

## 🖥️ Usage Guide

1. **Upload Data**  
   - Click "Browse files" in sidebar
   - Upload CSV with `smiles` column

2. **Configure Options**  
   - Select number of top compounds to include

3. **Run Screening**  
   - Click "🚀 Start Prediction & Screening"

4. **Analyze Results**  
   - Interactive results table
   - Visualizations preview
   - Download options:
     - CSV of results
     - Full PDF report

---

## 🗂 Project Structure

```
project-root/
├── app.py                      # Main application
├── saved_model/                # Model storage
│   ├── best_model.h5           # Trained model
│   └── feature_selector.pkl    # Feature selector
├── requirements.txt            # Dependencies
└── README.md                   # Documentation
```

---

## 🧠 Model Details

- **Fingerprints**: Morgan/ECFP (Radius 3, 2048 bits)
- **Feature Selection**: Pre-trained SelectFromModel
- **Note**: Must match training parameters

---

## 🤝 Contributing

Contributions welcome! Please:
1. Open an issue to discuss changes
2. Fork the repository
3. Submit a pull request

---

## 📜 License

[Specify your license here]
```

Key improvements made:
1. Better visual hierarchy with clear sections
2. More organized technology stack presentation
3. Improved installation instructions with proper code blocks
4. Added placeholder for license information
5. Better file structure visualization
6. More concise feature descriptions
7. Added proper screenshot placeholder
8. Consistent formatting throughout
9. Clearer contribution guidelines
10. Better table formatting for tech stack

Remember to:
1. Replace the placeholder screenshot link with actual app screenshots
2. Update the GitHub repository links
3. Add your actual license information
4. Include any additional credits or acknowledgments if needed
