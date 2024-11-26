# Predicting Student Success with Neural Networks

## Objective
A neural network model to predict whether students succeed in introductory business courses, developed as part of an MBA course on data mining.

## Problem Statement
The goal of this project was to assist university leadership in identifying at-risk students by developing a machine learning model to classify students as "successful" or "not successful" based on course and demographic data.

## Methodology
1. **Data Preparation**:
   - Cleaned and encoded categorical variables.
   - Split data into:
     - **Training Set (60%)**: For model training.
     - **Validation Set (20%)**: For tuning and avoiding overfitting.
     - **Test Set (20%)**: For final evaluation.

2. **Neural Network Development**:
   - Built using TensorFlow and Keras in Python.
   - Model architecture:
     - **Input Layer**: Features such as course repetitions, number of additional courses, and full-time work status.
     - **Two Hidden Layers**: Applied ReLU activation functions for non-linearity.
     - **Output Layer**: Used Sigmoid activation for binary classification.

3. **Performance Evaluation**:
   - Evaluated using metrics such as accuracy, precision, recall, F1-score, and ROC AUC.

## Results
- **Accuracy**: 72%
- **ROC AUC**: 0.74
- **Key Visuals**:
  - **Confusion Matrix**:
    ![Confusion Matrix](Confusion_Matrix.png)
  - **Classification Report**:
    ![Classification Report](Classification_Report.png)
  - **Model Accuracy Over Epochs**:
    ![Model Accuracy Over Epochs](Model_Accuracy_Over_Epochs.png)
  - **ROC Curve**:
    ![ROC Curve](ROC_Curve.png)

### Insights:
- The model is effective in identifying students who are at risk of not succeeding.
- Opportunities for improvement exist in predicting students who are likely to succeed (evidenced by recall metrics).

## Tools
- **Programming Languages and Libraries**:
  - Python 3.8+
  - TensorFlow 2.6+
  - Scikit-learn 0.24+
  - Matplotlib 3.4+
  - Pandas 1.3+
- **Dataset**:
  - CourseDFWI.csv (an anonymized dataset provided as part of the project).

## Dependencies
To replicate the project, ensure you have the following installed:
- TensorFlow
- Scikit-learn
- Matplotlib
- Pandas
- Jupyter Notebook (optional, for running the notebook interactively)

## Future Enhancements
- Incorporate additional features such as attendance or participation metrics to improve model accuracy.
- Experiment with ensemble methods like Random Forest or Gradient Boosting to enhance performance.
- Fine-tune hyperparameters for better recall and overall predictive power.

## Disclaimer
This project was completed as part of an MBA data mining course and is for educational purposes only.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/neural-network-student-success.git

2. Navigate into the project directory
   cd neural-network-student-success
3. Install dependencies:

Ensure you have Python 3.8+ installed.
Use the pip command to install the required libraries:
bash
Copy code
