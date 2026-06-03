# Heart Disease Prediction Using Machine Learning

## Project Overview
This project predicts the likelihood of heart disease using a Decision Tree Classifier trained on the UCI Cleveland Heart Disease dataset. It includes data preprocessing, feature scaling, model training, evaluation, feature importance analysis, and prediction.

## Features
- Data Loading and Cleaning
- Missing Value Handling
- Feature Scaling using StandardScaler
- Decision Tree Classification
- Model Evaluation
- Confusion Matrix Visualization
- Feature Importance Analysis
- Heart Disease Prediction
- Model Saving using Joblib

## Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn
- Joblib

## Project Structure

```text
Heart_Disease_Project/
│
├── project.py
├── heart.csv
├── README.md
├── heart_model.pkl
├── scaler.pkl
└── .gitignore
```

## Installation

Install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

## Run the Project

```bash
python project.py
```

## Dataset
The project uses the UCI Cleveland Heart Disease Dataset.

### Input Features
- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise-Induced Angina
- ST Depression
- Slope
- Number of Major Vessels
- Thalassemia

### Target
- 0 = No Heart Disease
- 1 = Heart Disease

## Machine Learning Workflow
1. Load Dataset
2. Handle Missing Values
3. Scale Features
4. Train Decision Tree Model
5. Evaluate Performance
6. Generate Confusion Matrix
7. Calculate Feature Importance
8. Predict Heart Disease Risk

## Sample Prediction

```python
sample = [63,1,3,145,233,1,0,150,0,2.3,0,0,1]
```

Example Output:

```text
Prediction: Heart Disease Detected
```

## Future Enhancements
- Random Forest Classifier
- XGBoost Model
- Hyperparameter Tuning
- Flask Web Application
- Model Deployment
- Interactive Dashboard

## Author
**Dugana Sai Manikanta**

GitHub: https://github.com/saidugana26-pixel

## License
This project is for educational and learning purposes.
