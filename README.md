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

###Requirements to run your Python code (libraries, etc)
- pip install numpy
- pip install pandas
- pip install matplotlib
- pip install seaborn
- pip install opencv-python
- pip install pillow
- pip install torch
- pip install torchvision
- pip install scikit-learn
- pip install joblib


🏋‍♂ Instructions to Train and Validate the Models
This project includes both Scikit-learn-based classical models and a PyTorch-based CNN. Below are the step-by-step instructions for running each model.

🔷 1. Classical Models (Scikit-learn)
Each model is implemented in a Jupyter Notebook. To train and validate them:

✅ Example: Random Forest
Open Random_Forest.ipynb in Jupyter or VS Code.

Run each cell in sequence:
Loads and splits the dataset.
Trains the Random Forest using GridSearchCV.
Evaluates on validation data using accuracy, precision, recall, F1-score.
The confusion matrix is displayed at the end.

Notebooks for Other Models:
Decision_tree_Supervised.ipynb
XG_Boosting.ipynb
Semi_DT.ipynb

🔷 2. CNN (PyTorch)
Files: Final_CNN.ipynb, cnn1.ipynb, best_model.pkl
✅ To Train the CNN:
Open Final_CNN.ipynb.

Make sure you have the Training/museum-indoor and Training/museum-outdoor folders as specified.
Run all cells:
Automatically performs train/val split
Trains CNN with different learning rates and epoch counts
Selects best model based on validation accuracy
Saves final model as best_model.pkl.

📦 Dependencies:
bash
pip install torch torchvision scikit-learn matplotlib seaborn
📈 To Validate the CNN on Test Data:
Open and run evaluate.py:

bash
python evaluate.py
This loads best_model.pkl, runs prediction on Museum_Validation/museum-indoor and museum-outdoor, and prints:

Accuracy, Precision, Recall, F1-score
Confusion matrix


🏋 Instructions to Train and Validate Your Models (CNN - PyTorch)
To train the CNN model from scratch and validate on your dataset:

✅ Requirements
Make sure the following Python packages are installed:

bash
Copy
Edit
pip install torch torchvision scikit-learn numpy matplotlib seaborn pillow
✅ Folder Structure
Ensure the training dataset is organized like:

Copy
Edit
/Training
├── museum-indoor/
└── museum-outdoor/
✅ Training Instructions
Open Final_CNN.ipynb using Jupyter Notebook or VS Code.

Run all cells sequentially:

Loads and labels images from both folders.

Splits into training and validation sets (80/20).

Trains a CNN using multiple hyperparameter combinations:

Learning rates: 0.01, 0.001, 0.0001

Epochs: 2 for tuning, 10 for final training

Prints accuracy per epoch and selects best model.

Saves best model to: best_model.pkl

🧪 Instructions to Run Pre-trained CNN on Test Data
You can evaluate the saved model (best_model.pkl) on new test images.

✅ Folder Structure
Ensure the test dataset is placed in:

Copy
Edit
/Museum_Validation
├── museum-indoor/
└── museum-outdoor/
✅ Running Evaluation
Run the evaluate.py script (you can create one based on your notebook logic):

bash
Copy
Edit
python evaluate.py
It will:

Load best_model.pkl

Predict on all test images from both folders

Print:

Accuracy

Precision

Recall

F1-score

Confusion Matrix (visualized with Seaborn)

📌 Example Output:
makefile
Copy
Edit
Accuracy: 0.9300
Precision: 0.9302
Recall: 0.9300
F1-score: 0.9300
Plus a confusion matrix plot will be displayed.


Source Code Package (Scikit-learn & PyTorch)
Our implementation consists of two main model pipelines: classical machine learning models using Scikit-learn, and deep learning models using PyTorch. Each pipeline is encapsulated in dedicated notebooks/scripts and includes complete support for data loading, model training, validation, and performance evaluation.

🔷 Scikit-learn Models
We implemented and evaluated the following classical models using Scikit-learn:

Model	File	Description
Decision Tree	Decision_tree_Supervised.ipynb	Supervised DT with grid search for max_depth, criterion
Random Forest	Random_Forest.ipynb	Ensemble of trees, tuned via n_estimators, max_depth
Gradient Boosting	XG_Boosting.ipynb	Boosting model using learning rate and early stopping
Semi-Supervised DT	Semi_DT.ipynb	Pseudo-labeling applied iteratively on unlabeled samples
Each notebook includes:

Data preprocessing and splitting

Model training with tuned hyperparameters

Evaluation using Accuracy, Precision, Recall, and F1-score

Confusion matrix visualization using seaborn

🔷 PyTorch CNN
Our deep learning approach is implemented using PyTorch and includes the following files:

File	Description
Final_CNN.ipynb	Main notebook containing CNN architecture, training/validation pipeline, and results
cnn1.ipynb	Early prototype of CNN used for experimentation
best_model.pkl	Saved trained CNN model (PyTorch state_dict)
evaluate.py	Loads best_model.pkl, runs prediction on validation set, prints classification report and confusion matrix
predict_image.py	Optional script to run inference on a single image using the saved CNN
CNN Architecture:

2–3 Convolutional blocks (Conv2D -> ReLU -> MaxPool)

Fully connected layers with dropout

Trained using Adam optimizer and CrossEntropyLoss

Tuned for different learning rates [0.01, 0.001, 0.0001] and epochs [10, 20]
