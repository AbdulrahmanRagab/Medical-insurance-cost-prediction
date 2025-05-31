
# 🏥 Insurance Cost Prediction - Machine Learning Project

## Table of Contents
- [🌟 Overview](#-overview)
- [🛠️ Technologies Used](#-technologies-used)
- [🎯 Project Objectives](#-project-objectives)
- [📊 Dataset Description](#-dataset-description)
- [🚀 Methodology](#-methodology)
- [🧠 Model Training and Evaluation](#-model-training-and-evaluation)
- [📈 Results](#-results)
- [🎉 Conclusion](#-conclusion)
- [🙏 Acknowledgments](#-acknowledgments)
- [🔮 Future Work](#-future-work)
- [🛠️ How to Use](#-how-to-use)

---

## 🌟 Overview
This project aims to predict **insurance costs** based on various factors such as age, BMI, smoking status, and region. The goal is to build a robust machine learning model that can accurately estimate insurance charges for individuals. The project involves data preprocessing, feature engineering, model selection, and evaluation using regression techniques.

---

## 🛠️ Technologies Used

### Programming Language
- **Python**  
  ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

### Frameworks and Libraries
- **Pandas** (for data manipulation)  
  ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
- **NumPy** (for numerical computations)  
  ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
- **Scikit-learn** (for machine learning)  
  ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
- **Matplotlib** (for data visualization)  
  ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)
- **Seaborn** (for data visualization)  
  ![Seaborn](https://img.shields.io/badge/Seaborn-29BEB0?style=for-the-badge&logo=seaborn&logoColor=white)
- **Plotly** (for interactive visualizations)  
  ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
- **Dash** (for building interactive dashboards)  
  ![Dash](https://img.shields.io/badge/Dash-008DE4?style=for-the-badge&logo=dash&logoColor=white)
- **Pickle** (for model serialization)  
  ![Pickle](https://img.shields.io/badge/Pickle-000000?style=for-the-badge&logo=pickle&logoColor=white)

---

## 🎯 Project Objectives
1. **📊 Data Preprocessing**: Clean and preprocess the dataset to handle missing values, encode categorical variables, and scale numerical features.
2. **🧠 Model Selection**: Experiment with various regression models (e.g., Linear Regression, Ridge, Lasso, Gradient Boosting, Random Forest) to identify the best-performing model.
3. **📈 Model Evaluation**: Evaluate models using metrics such as RMSE (Root Mean Squared Error) and R² score.
4. **🚀 Deployment**: Save the best model using Pickle for future predictions and build an interactive dashboard using Dash.

---

## 📊 Dataset Description
The dataset used in this project is **insurance.csv**, which contains the following features:
- **age**: Age of the individual.
- **sex**: Gender of the individual (male/female).
- **bmi**: Body Mass Index (BMI) of the individual.
- **children**: Number of children/dependents.
- **smoker**: Smoking status (yes/no).
- **region**: Region of residence (northeast, northwest, southeast, southwest).
- **charges**: Insurance charges (target variable).

---

## 🚀 Methodology

### 1. **Data Preprocessing**
   - **📂 Data Loading**: The dataset was loaded using Pandas.
   - **🔍 Data Cleaning**:
     - Negative values in the `age` column were converted to positive values.
     - Missing values in numerical columns (`age`, `bmi`, `charges`) were imputed with the mean.
     - Missing values in the `children` column were imputed with the median.
   - **🔧 Feature Engineering**:
     - Categorical variables (`sex`, `smoker`, `region`) were encoded using `OneHotEncoder`.
     - Numerical features (`age`, `bmi`) were scaled using `StandardScaler`.

### 2. **Exploratory Data Analysis (EDA)**
   - **📊 Visualizations**:
     - Distribution of categorical variables (`sex`, `smoker`, `region`) using bar plots and pie charts.
     - Distribution of numerical variables (`age`, `bmi`, `charges`) using histograms, box plots, and scatter plots.
     - Pair plots to visualize relationships between numerical features.
   - **🔍 Outlier Detection**:
     - Outliers in `age`, `bmi`, and `charges` were identified using the Interquartile Range (IQR) method.

### 3. **Model Selection**
   - **🧠 Models Tested**:
     - **Linear Regression**
     - **Ridge Regression**
     - **Lasso Regression**
     - **Gradient Boosting Regressor**
     - **Random Forest Regressor**
   - **🛠️ Hyperparameter Tuning**: GridSearchCV was used to find the best hyperparameters for each model.

### 4. **Model Evaluation**
   - Models were evaluated using **RMSE** and **R² score**.
   - The best-performing model was saved using Pickle for future predictions.

---

## 🧠 Model Training and Evaluation

### 1. **Gradient Boosting Regressor**
   - **Best Parameters**:
     ```python
     {'grad_boosting_model__learning_rate': 0.1, 'grad_boosting_model__n_estimators': 60}
     ```
   - **RMSE**: 4,532.12
   - **R² Score**: 0.87 (Training), 0.86 (Testing)

### 2. **Linear Regression**
   - **RMSE**: 5,789.34
   - **R² Score**: 0.78 (Testing)

### 3. **Polynomial Regression with Lasso**
   - **Best Parameters**:
     ```python
     {'lasso_model__alpha': 0.1, 'poly__degree': 2}
     ```
   - **RMSE**: 4,890.45
   - **R² Score**: 0.84 (Testing)

### 4. **Random Forest Regressor**
   - **Best Parameters**:
     ```python
     {'random_model__max_depth': None, 'random_model__min_samples_leaf': 1, 'random_model__min_samples_split': 2, 'random_model__n_estimators': 66}
     ```
   - **RMSE**: 4,210.56
   - **R² Score**: 0.88 (Testing)

---

## 📈 Results
- **Best Model**: **Gradient Boosting Regressor** achieved the highest R² score (0.86) and the lowest RMSE (4,532.12) on the test set.
- **Model Serialization**: The best model was saved using Pickle for future use.

---

## 🎉 Conclusion
This project successfully built a machine learning model to predict insurance costs with high accuracy. The **Gradient Boosting Regressor** outperformed other models, demonstrating its effectiveness for this regression task. The model can be further improved by incorporating additional features or experimenting with more advanced algorithms.

---

## 🙏 Acknowledgments
- **📚 Scikit-learn Documentation**: For providing comprehensive guidance on machine learning models.
- **🤖 Kaggle Community**: For sharing datasets and valuable insights.

---

## 🔮 Future Work
- **📊 Feature Expansion**: Incorporate additional features such as medical history and lifestyle factors.
- **🧠 Advanced Models**: Experiment with deep learning models like Neural Networks.
- **🌐 Deployment**: Deploy the model as a web application for real-time predictions.

---

## 🛠️ How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/AbdulrahmanRagab/insurance-cost-prediction.git
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook or Python script to train and evaluate the models:
   ```bash
   jupyter notebook Insurance_Cost_Prediction.ipynb
   ```
4. Load the saved model for predictions:
   ```python
   import pickle
   model = pickle.load(open("GBR_model.pkl", "rb"))
   predictions = model.predict(new_data)
   ```
