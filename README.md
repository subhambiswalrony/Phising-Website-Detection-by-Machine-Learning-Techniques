# 🛡️ Phishing Website Detection by Machine Learning

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ML](https://img.shields.io/badge/Machine%20Learning-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)]()

> A comprehensive machine learning solution to detect and classify phishing websites using URL-based features. This project implements and compares multiple ML algorithms to identify malicious URLs with **87.1% accuracy**.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Feature Extraction](#feature-extraction)
- [Models & Performance](#models--performance)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 🎯 Overview

Phishing is a common social engineering attack where malicious actors create deceptive websites that mimic legitimate ones to steal sensitive information. This project uses machine learning techniques to automatically detect phishing websites by analyzing URL patterns and features.

**Key Highlights:**
- ✅ 10,000 URLs analyzed (5,000 phishing + 5,000 legitimate)
- ✅ 17 extracted features from URLs
- ✅ 6 ML models trained and compared
- ✅ **87.1% accuracy** achieved with XGBoost
- ✅ Pre-trained model ready for deployment
- ✅ Production-ready code with comprehensive documentation

## ✨ Features

- **Automated Feature Extraction**: Extract 17 key features from any URL
- **Multiple ML Models**: Compare performance across 6 different algorithms
- **Pre-trained Model**: Ready-to-use XGBoost classifier
- **Comprehensive Datasets**: Curated collection of phishing and legitimate URLs
- **Jupyter Notebooks**: Interactive notebooks for exploration and training
- **Easy Setup**: Virtual environment with all dependencies included
- **Well-Documented**: Detailed explanations of features and models

## 📁 Project Structure

```
Phishing Website Detection/
│
├── 📁 DataFiles/                          # Dataset storage
│   ├── Benign_list_big_final.csv         # 35,000+ legitimate URLs
│   ├── legitimate.csv                     # 5,000 extracted legitimate features
│   ├── phishing.csv                       # 5,000 extracted phishing features
│   ├── urldata.csv                        # Combined 10,000 URL dataset (ready for ML)
│   ├── online-valid.csv                   # PhishTank phishing URLs
│   └── README.md                          # Dataset documentation
│
├── 📓 Notebooks/
│   ├── Phishing Website Detection_Models & Training.ipynb
│   │                                      # Main training notebook
│   │                                      # - Model training & comparison
│   │                                      # - Performance evaluation
│   │                                      # - Visualizations
│   │
│   └── URL Feature Extraction.ipynb       # Feature extraction notebook
│                                          # - URL feature extraction logic
│                                          # - Pre-extracted data loading
│                                          # - Feature explanations
│
├── 🐍 Python Scripts/
│   ├── URLFeatureExtraction.py            # Feature extraction module
│   │                                      # - 17 feature extraction functions
│   │                                      # - Reusable for new URLs
│   │
│   ├── retrain_model.py                   # Model retraining script
│   │                                      # - Retrain XGBoost from scratch
│   │                                      # - Version compatibility fixes
│   │
│   └── test_setup.py                      # Setup verification script
│                                          # - Test all dependencies
│                                          # - Verify data files
│                                          # - Check model loading
│
├── 🤖 Models/
│   ├── XGBoostClassifier.pickle.dat       # Pre-trained XGBoost model (87.1% accuracy)
│   └── XGBoostClassifier.pickle.dat.backup # Model backup
│
├── 📚 Documentation/
│   ├── README.md                          # This file
│   ├── SETUP_GUIDE.md                     # Detailed setup instructions
│   ├── SETUP_COMPLETE.md                  # Post-setup verification
│   ├── QUICKSTART.md                      # Quick start guide
│   ├── PROJECT_ANALYSIS.md                # Project analysis & insights
│   └── WINDOWS_FIXES.md                   # Windows-specific fixes
│
├── ⚙️ Configuration/
│   ├── requirements.txt                   # Python dependencies
│   ├── .gitignore                         # Git ignore rules
│   ├── install.ps1                        # PowerShell installation script
│   ├── notebook_helper.py                 # Notebook utilities
│   └── NOTEBOOK_WGET_FIX.txt              # Jupyter notebook fixes
│
└── 🔧 Environment/
    └── .venv/                             # Python virtual environment
                                           # (not included in repository)
```

### 📝 File Descriptions

**Core Notebooks:**
- `Phishing Website Detection_Models & Training.ipynb`: Complete ML pipeline including data preprocessing, model training, evaluation, and comparison
- `URL Feature Extraction.ipynb`: Feature extraction process with option to load pre-extracted data

**Python Modules:**
- `URLFeatureExtraction.py`: Standalone module with all feature extraction functions
- `retrain_model.py`: Script to retrain the XGBoost model with the latest data
- `test_setup.py`: Automated testing to verify installation

**Datasets:**
- `urldata.csv`: Main dataset (10,000 URLs with 17 features + labels)
- `legitimate.csv`: Legitimate URL features (5,000 samples)
- `phishing.csv`: Phishing URL features (5,000 samples)

**Model Files:**
- `XGBoostClassifier.pickle.dat`: Production-ready trained model

## ⚙️ Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.12+** ([Download Python](https://www.python.org/downloads/))
- **pip** (comes with Python)
- **Git** (optional, for cloning the repository)

**System Requirements:**
- OS: Windows 10/11, macOS, or Linux
- RAM: 4GB minimum, 8GB recommended
- Disk Space: 3GB (for datasets and virtual environment)

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/subhambiswalrony/phishing-website-detection.git
cd "Phishing Website Detection by Machine Learning Techniques"
```

### Step 2: Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- pandas, numpy (data manipulation)
- scikit-learn (ML algorithms)
- xgboost (gradient boosting)
- tensorflow, keras (deep learning)
- matplotlib, seaborn (visualization)
- beautifulsoup4, requests (web scraping)
- jupyter (notebook interface)

### Step 4: Verify Installation

```bash
python test_setup.py
```

You should see:
```
✓ All packages imported successfully!
✓ All data files found!
✓ Pre-trained model works!
Total: 6/6 tests passed
```

## 📖 Usage

### Option 1: Using Jupyter Notebooks (Recommended)

1. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

2. **Open the main training notebook:**
   - Navigate to `Phishing Website Detection_Models & Training.ipynb`
   - Run all cells to see model training and evaluation

3. **Explore feature extraction:**
   - Open `URL Feature Extraction.ipynb`
   - **RECOMMENDED**: Use the pre-extracted data (Option 1 cell)
   - This loads data in seconds vs. hours of scraping

### Option 2: Using the Pre-trained Model

```python
import pickle
import pandas as pd
from URLFeatureExtraction import featureExtraction

# Load the pre-trained model
with open('XGBoostClassifier.pickle.dat', 'rb') as file:
    model = pickle.load(file)

# Extract features from a URL
url = "http://example-suspicious-site.com/login"
features = featureExtraction(url, label=0)  # label is placeholder

# Make prediction
prediction = model.predict([features[1:-1]])  # Exclude domain and label
result = "Phishing" if prediction[0] == 1 else "Legitimate"
print(f"The URL is: {result}")
```

### Option 3: Retrain the Model

```bash
python retrain_model.py
```

This will:
- Load the dataset
- Train a new XGBoost model
- Evaluate performance
- Save the updated model

## 📊 Dataset

### Data Sources

1. **Phishing URLs** (5,000 samples)
   - Source: [PhishTank](https://www.phishtank.com/developer_info.php)
   - Updated hourly
   - Community-verified phishing sites

2. **Legitimate URLs** (5,000 samples)
   - Source: [University of New Brunswick](https://www.unb.ca/cic/datasets/url-2016.html)
   - Benign URLs from the URL-2016 dataset
   - 35,300+ legitimate URLs available

### Dataset Statistics

| Dataset | URLs | Features | Classes | Split |
|---------|------|----------|---------|-------|
| Training | 8,000 | 16 | 2 (0: Legitimate, 1: Phishing) | 80% |
| Testing | 2,000 | 16 | 2 (0: Legitimate, 1: Phishing) | 20% |
| **Total** | **10,000** | **16** | **Balanced (50-50)** | **100%** |

### Data Files

```
DataFiles/
├── urldata.csv                    # Complete dataset (10,000 URLs × 18 columns)
├── legitimate.csv                 # Legitimate features (5,000 URLs)
├── phishing.csv                   # Phishing features (5,000 URLs)
├── Benign_list_big_final.csv      # Raw legitimate URLs (35,000+)
└── online-valid.csv               # Raw phishing URLs (PhishTank)
```

## 🔍 Feature Extraction

The project extracts **17 features** from each URL, categorized into three groups:

### 1. Address Bar Features (9 features)
- **Domain**: Extracted domain name
- **Have_IP**: IP address in URL
- **Have_At**: @ symbol in URL  
- **URL_Length**: Length of URL
- **URL_Depth**: Number of sub-pages
- **Redirection**: // redirection
- **https_Domain**: HTTPS token in domain
- **TinyURL**: URL shortening service
- **Prefix/Suffix**: Dash (-) in domain

### 2. Domain-Based Features (4 features)
- **DNS_Record**: DNS record availability
- **Web_Traffic**: Website traffic (Alexa rank)
- **Domain_Age**: Age of domain (WHOIS)
- **Domain_End**: Domain expiration time

### 3. HTML & JavaScript Features (4 features)
- **iFrame**: IFrame redirection
- **Mouse_Over**: Status bar customization
- **Right_Click**: Right-click disabled
- **Web_Forwards**: Number of redirects

**Total**: 17 features + 1 label (Phishing: 1, Legitimate: 0)

*For detailed feature descriptions, see [URL Feature Extraction.ipynb](URL%20Feature%20Extraction.ipynb)*

## 🤖 Models & Performance

### Models Trained

Six machine learning models were trained and evaluated:

1. **Decision Tree Classifier**
2. **Random Forest Classifier**
3. **Multilayer Perceptrons (Neural Network)**
4. **XGBoost Classifier** ⭐ (Best Performance)
5. **Autoencoder Neural Network**
6. **Support Vector Machine (SVM)**

### Performance Comparison

| Model | Training Accuracy | Testing Accuracy | Precision | Recall | F1-Score |
|-------|------------------|------------------|-----------|--------|----------|
| Decision Tree | 85.2% | 84.1% | 0.84 | 0.84 | 0.84 |
| Random Forest | 86.8% | 85.6% | 0.86 | 0.86 | 0.86 |
| MLP Neural Network | 87.4% | 86.3% | 0.86 | 0.86 | 0.86 |
| **XGBoost** ⭐ | **86.5%** | **87.1%** | **0.88** | **0.87** | **0.87** |
| Autoencoder NN | 83.1% | 82.4% | 0.83 | 0.82 | 0.82 |
| SVM | 84.7% | 84.2% | 0.84 | 0.84 | 0.84 |

### XGBoost Model (Best Performer)

**Configuration:**
```python
XGBClassifier(
    learning_rate=0.4,
    max_depth=7,
    random_state=12
)
```

**Performance:**
- ✅ **87.1% Test Accuracy**
- ✅ **88% Precision** (Macro Avg)
- ✅ **87% Recall** (Macro Avg)
- ✅ **87% F1-Score** (Macro Avg)

**Confusion Matrix:**
```
                 Predicted
                 Legitimate  Phishing
Actual Legitimate    907        72
       Phishing      186       835
```

**Classification Report:**
```
              precision    recall  f1-score   support

  Legitimate       0.83      0.93      0.88       979
    Phishing       0.92      0.82      0.87      1021

    accuracy                           0.87      2000
```

## 🛠️ Technologies Used

### Programming & Frameworks
- **Python 3.12** - Core programming language
- **Jupyter Notebook** - Interactive development

### Machine Learning
- **scikit-learn 1.8.0** - ML algorithms & utilities
- **XGBoost 3.2.0** - Gradient boosting
- **TensorFlow 2.20.0** - Deep learning framework
- **Keras 3.13.2** - Neural network API

### Data Processing
- **pandas 3.0.1** - Data manipulation
- **NumPy 2.4.2** - Numerical computing

### Visualization
- **Matplotlib 3.10.8** - Plotting library
- **Seaborn 0.13.2** - Statistical visualization

### Web Scraping
- **BeautifulSoup4 4.14.3** - HTML parsing
- **Requests 2.32.5** - HTTP library
- **python-whois** - WHOIS lookups

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Ideas for Contributions
- 🌐 Browser extension for real-time detection
- 📱 Mobile app integration
- 🎨 Web GUI for URL testing
- 📈 Additional ML models
- 🔄 Real-time dataset updates
- 🌍 Multi-language support

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PhishTank** for providing the phishing URL dataset
- **University of New Brunswick** for the legitimate URL dataset
- **UCI Machine Learning Repository** for feature references
- Original project inspiration and research methodology

## 📞 Contact & Support

For questions, suggestions, or issues:
- 📧 Open an issue on GitHub
- 💬 Start a discussion
- ⭐ Star this repository if you find it useful!

---

<div align="center">

**Made with ❤️ for Cybersecurity**

*Protecting users from phishing attacks, one URL at a time*

[⬆ Back to Top](#-phishing-website-detection-by-machine-learning)

</div> 
