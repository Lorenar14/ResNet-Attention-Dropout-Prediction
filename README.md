
#  Design of an Early Warning System Based on ResNet and Attention Mechanisms for University Dropout Prediction: A Case Study of Politécnico Colombiano Jaime Isaza Cadavid in Colombia

Deep learning framework for university student dropout prediction using a ResNet-Attention-based Early Warning System and institutional data from Politécnico Colombiano Jaime Isaza Cadavid (Colombia). The proposed methodology integrates data preprocessing, exploratory data analysis, survival probability analysis, class balancing using SMOTE, hyperparameter optimization with Optuna, and a ResNet-Attention deep learning architecture as the main predictive model. Logistic Regression, Random Forest, and XGBoost were implemented as baseline models for comparative performance evaluation.


SCRIPTS: 
1_DataPreprocessing.ipynb: Performs data cleaning, anonymization, preprocessing, feature engineering, and target variable construction.

2_ExploratoryDataAnalysis.ipynb: Generates descriptive statistics and exploratory visualizations, including the correlation matrix, pairplot, boxplots, and survival probability analysis.

3_SMOTEBalancing.ipynb: Applies the SMOTE (Synthetic Minority Over-sampling Technique) to balance the training dataset.

4_ResNetAttentionModel.ipynb: Implements, trains, and optimizes the proposed ResNet-Attention deep learning model for university dropout prediction.

5_BaselineModels.ipynb: Implements Logistic Regression, Random Forest, and XGBoost as baseline models for performance comparison.

6_ModelEvaluation.ipynb: Evaluates model performance using Accuracy, Precision, Recall, F1-Score, ROC Curve, AUC-ROC, and Confusion Matrix.

7_DropoutPrediction.ipynb: Applies the trained ResNet-Attention model to predict the dropout risk of new university students.

MODELS: The trained ResNet-Attention model and the baseline Machine Learning models are stored in the Models folder.

IMPORTANT: Update the dataset paths before running the scripts. The original institutional dataset is not publicly available due to privacy and confidentiality restrictions.

CITATION: These scripts and trained models were developed to obtain the results presented in the paper: Design of an Early Warning System Based on ResNet and Attention Mechanisms for University Dropout Prediction: A Case Study of Politécnico Colombiano Jaime Isaza Cadavid in Colombia.
