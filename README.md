
# 🔬 Bioinformatics Disease Classification with GNNs & Feature Selection

A comprehensive bioinformatics pipeline leveraging **Graph Neural Networks (GNNs)**, **feature selection**, and traditional machine learning techniques to classify diseases from gene expression data. Built using **Python, PyTorch Geometric, Scikit-learn**, and supporting libraries for robust analysis and visualization.

---

## 🚀 Tech Stack
- **Python 3.x**
- **PyTorch & PyTorch Geometric** – for GNN architectures (GCN, GAT)
- **Scikit-learn** – feature selection (ANOVA F-value, Chi-Square), traditional ML models
- **Pandas, NumPy** – data processing
- **Matplotlib** – visualization
- **Jupyter Notebook** – workflow management

---

## 📂 Project Structure
```
CAP5510-Bioinformatics-Project/
├── analysis.ipynb                # Visualization & PCA analysis
├── extract_features_100to500.ipynb  # Feature extraction & selection
├── featureSelection.ipynb        # Advanced feature selection techniques
├── gnn.ipynb                     # GNN-based classification (GCN)
├── gnn_all_features.ipynb        # GNN with all features (GCN, GAT)
├── knn.ipynb, naiveBayes.ipynb, randomForest.ipynb  # Traditional ML models
├── datasets/                     # Raw GEO datasets (GSE4290, GSE19804, etc.)
├── preprocessed/                 # Preprocessed datasets (filtered, standardized)
├── model_results.csv             # Evaluation metrics
└── results.ipynb                 # Final analysis and comparison
```

---

## ⚙️ Setup Instructions
1️⃣ **Clone the repository**
```bash
git clone https://github.com/your-username/CAP5510-Bioinformatics-Project.git
cd CAP5510-Bioinformatics-Project
```

2️⃣ **Create and activate a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

3️⃣ **Install dependencies**
```bash
pip install -r requirements.txt
```
*Dependencies include: torch, torch-geometric, scikit-learn, pandas, numpy, matplotlib.*

4️⃣ **Run Jupyter Notebooks**
```bash
jupyter notebook
```
Open notebooks such as `gnn.ipynb`, `featureSelection.ipynb`, and `analysis.ipynb`.

---

## 📈 Key Features
- 🔬 Preprocessing of bioinformatics datasets (GSE series) with dimensionality reduction.
- 🏷️ Feature extraction and selection using ANOVA F-value, Chi-Square, and custom methods.
- 🤖 Classification with multiple models: Random Forest, Naive Bayes, KNN.
- 🌐 Advanced GNN models (GCN, GAT) with PyTorch Geometric for disease classification.
- 📊 Visualization of results (PCA, accuracy/loss plots, performance comparison).

---

## 📊 Results
✅ Achieved disease classification accuracy with dimensionality reduction and graph-based learning.  
✅ Validated performance on multiple GEO datasets (e.g., GSE4290, GSE19804, GSE27562).

---

## 🤝 Contributing
Contributions are welcome! Feel free to fork this repository and submit pull requests to enhance data processing, model performance, or add new features.

---

