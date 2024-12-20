# GAN-BERT for Sentiment Analysis

## Overview
**GAN-BERT** is a hybrid model that combines Generative Adversarial Networks (GANs) with the BERT Transformer architecture for semi-supervised text classification. This repository includes scripts for training a teacher-student model using knowledge distillation, as well as visualizing data analysis and model performance

## Key Features
- **Generative Adversarial Networks (GANs)** for enhanced data generation in semi-supervised learning.
- **BERT-based Teacher Model** for high-performance pre-trained language understanding.
- **Student Model with Knowledge Distillation** to transfer knowledge from the teacher to a lightweight model.
- **Data Analysis** to explore and visualize the dataset prior to model training.
- **Visualizations** to evaluate model performance and data characteristics.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Requirements](#requirements)
## Scripts to run

### Running scripts
pip install -r requirements.txt

python scripts/train_teacher.py
python scripts/train_ganbert.py
python scripts/data_analysis.py


---



### code structure
```bash




ganbert-text-classification/
├── data/                   # Dataset folder
│   ├── train.csv
│   ├── valid.csv
│   └── test.csv
├── models/                 # Model architecture files
│   ├── generator.py        # Generator model
│   ├── discriminator.py    # Discriminator model
│   ├── teacher_model.py    # Teacher model (BERT-based)
│   └── student_model.py    # Student model (LSTM/GRU-based)
├── utils/                  # Utility scripts
│   ├── data_loader.py      # Data loading functions
│   ├── preprocessing.py    # Text preprocessing functions
│   ├── metrics.py          # Evaluation metrics
│   └── attention.py        # Self-attention mechanism
├── scripts/                # Training and analysis scripts
│   ├── train_ganbert.py    # GAN-BERT training
│   ├── train_teacher.py    # Teacher training
│   ├── train_student.py    # Student training
│   └── data_analysis.py    # Data analysis script
├── notebooks/              # Jupyter notebooks for exploration
│   └── exploratory_analysis.ipynb
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
└── main.py                 # Entry point script





pip install tensorflow torch transformers pandas numpy matplotlib seaborn scikit-learn missingno datasets


pip install -r requirements.txt


---

### **Explanation of Sections:**

1. **Overview**: 
   - Describes the **GAN-BERT** project, explaining the Teacher model (BERT-based), the Student model (LSTM or GRU), and how they interact within the **GAN-BERT** framework.

2. **Key Features**: 
   - Lists the major features of the project, including the use of GANs for semi-supervised learning, knowledge distillation, and the lightweight Student models (LSTM/GRU).

3. **Table of Contents**: 
   - Provides quick navigation to different sections in the README.

4. **Installation**:
   - Instructions for cloning the repository and installing dependencies using `pip` via the `requirements.txt` file.
   - Also provides instructions for manual installation of dependencies.

5. **Usage**:
   - Step-by-step instructions to train the **Teacher**, **Student**, and **GAN-BERT** models.
   - Also includes how to run the **data analysis** script.

6. **Directory Structure**:
   - A clear overview of the folder structure of the repository, detailing where each component resides (e.g., `models/`, `scripts/`, `data/`, etc.).

7. **Requirements**:
   - Lists all Python libraries required to run the project and provides installation instructions.


8. **Acknowledgements**:
   - Credits to **Hugging Face** and other contributors.

---


