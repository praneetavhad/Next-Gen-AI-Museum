Praneet Avhad - 40279347  
Rolwyn Raju - 40303902  
Pretty Kotian - 40320837



## Project Title: Museum Image Classification

### Overview
This project addresses the problem of classifying museum images into indoor and outdoor categories, a fundamental task in computer vision with practical applications in robotics, scene understanding, and AI perception systems. Using the Places MIT dataset, our team developed and evaluated a suite of machine learning models including:

- Decision Tree (Supervised)
- Random Forest (Supervised)
- Boosting (Gradient/XGBoost)
- Semi-Supervised Decision Tree (Pseudo-labeling strategy)
- Convolutional Neural Network (CNN) built from scratch using PyTorch

### Objective
Our goal was to compare traditional tree-based models and semi-supervised approaches with a deep learning-based CNN to assess classification performance on a binary image dataset. We explored various hyperparameters for each model to optimize performance and ensure fair evaluation using metrics such as accuracy, precision, recall, F1-score, and confusion matrix.

### Highlights
- **Random Forest** achieved 79.5% accuracy using 300 estimators.
- **Boosting** reached 82.5% accuracy after reducing depth and estimators to prevent overfitting.
- **Semi-Supervised DT** reached 73.5%, limited by pseudo-labeling noise.
- **CNN**, with optimized learning rate and architecture, achieved the best result of **90.5% accuracy**.

### Tools & Libraries
- PyTorch (for CNN)
- Scikit-learn (for classical ML models and evaluation)
- NumPy, Matplotlib, Seaborn

### Contributions
Each team member contributed to specific phases: data preparation and preprocessing, model training, hyperparameter tuning, performance evaluation, and result visualization.

### Dataset
We used the `Places MIT` museum subset, comprising:
- **8119 indoor images**
- **4221 outdoor images**

Each image was resized, normalized, and used across training, validation, and testing phases in a fully automated pipeline.

### Future Work
Further improvements include adopting deeper CNN architectures (e.g., ResNet), hybrid approaches combining CNNs with ensemble classifiers, and refining pseudo-labeling in semi-supervised learning.

