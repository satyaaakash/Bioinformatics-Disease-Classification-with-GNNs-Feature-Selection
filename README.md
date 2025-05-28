# 🔬 Bioinformatics Disease Classification with GNNs & Feature Selection

A comprehensive bioinformatics pipeline leveraging **Graph Neural Networks (GNNs)**, **feature selection**, and traditional machine learning techniques to classify diseases from gene expression data. Built using **Python, PyTorch Geometric, Scikit-learn**, and supporting libraries for robust analysis and visualization.

---

## 🚀 **Tech Stack**
- **Python 3.x**
- **PyTorch & PyTorch Geometric** (for GNN architectures)
- **Scikit-learn** (feature selection, ML models)
- **Pandas, NumPy** (data processing)
- **Matplotlib** (visualization)
- **Jupyter Notebook** (workflow)

---

## 📂 **Project Structure**
CAP5510-Bioinformatics-Project/
├── analysis.ipynb # Visualization & PCA analysis
├── extract_features_100to500.ipynb # Feature extraction & selection
├── featureSelection.ipynb # Advanced feature selection techniques
├── gnn.ipynb # GNN-based classification (GCN)
├── gnn_all_features.ipynb # GNN with all features (GCN, GAT)
├── knn.ipynb, naiveBayes.ipynb, randomForest.ipynb # ML models
├── datasets/ # Raw GEO datasets (GSE4290, GSE19804, etc.)
├── preprocessed/ # Preprocessed datasets (filtered, standardized)
├── model_results.csv # Evaluation metrics
└── results.ipynb # Final analysis and comparison


---

## ⚙️ **Setup Instructions**
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/CAP5510-Bioinformatics-Project.git
   cd CAP5510-Bioinformatics-Project
Create a virtual environment and activate it:
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:
pip install -r requirements.txt

Run Jupyter notebooks for specific workflows:
jupyter notebook



📈 Key Features
Preprocesses bioinformatics datasets (GSE series) with dimensionality reduction.

Extracts and selects optimal features using ANOVA F-value, Chi-Square, and custom techniques.

Implements multiple classification models: Random Forest, Naive Bayes, KNN.

Builds Graph Neural Networks (GCN, GAT) for disease classification with PyTorch Geometric.

Visualizes results using PCA, accuracy/loss plots, and comparison charts.



📊 Results
Achieved robust disease classification with dimensionality reduction and graph-based learning.

Validated performance on multiple GEO datasets (e.g., GSE4290, GSE1980


🤝 Contributing
Contributions are welcome! Feel free to fork this repository and submit pull requests to improve data processing, model performance, or add new features.
