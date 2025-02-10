# Predicitive-Analysis-of-Chichago-Traffic-Crashes: Crash Types and Damage Estimation  

This project focuses on analyzing traffic crashes in Chicago to predict crash types, estimate damage severity, and identify risk factors contributing to severe accidents. The study utilized advanced data analytics, machine learning techniques, distributed computing with Apache Spark, and a user-friendly web interface to uncover insights that can enhance urban traffic safety management.  

## Phase 1: Data Cleaning and Exploratory Data Analysis (EDA)  
Extensive data cleaning and preprocessing steps were performed on a traffic crash dataset from June to December 2023. Key tasks included:  
- Handling missing values  
- Resolving inconsistencies  
- Encoding categorical features  

Exploratory Data Analysis (EDA) was carried out to uncover trends and correlations, forming the foundation for predictive modeling.  

## Phase 2: Machine Learning for Prediction and Classification  
Multiple machine learning models were applied to predict crash types and assess contributing factors. Models included:  
- Logistic Regression  
- Random Forest  
- Support Vector Machine (SVM)  
- XGBoost  

Among these, XGBoost emerged as the best-performing model with an accuracy of 89.3% and an AUC score of 0.95, making it the most suitable choice for real-world applications.  

## Phase 3: Distributed Data Processing with PySpark  
The final phase utilized PySpark to implement all data preprocessing and machine learning tasks in a distributed environment. This approach provided:  
- Efficient handling of large datasets with over 1.5 million rows  
- Scalability and fault tolerance  
- Significant reductions in execution time for model training and evaluation  

PySpark's distributed machine learning pipelines and advanced analytics capabilities showcased its robustness for big data workflows.  

## Web User Interface  
A user-friendly web interface was developed to make the insights and predictions accessible to a broader audience. This interface allows users to:  
- Upload datasets for analysis  
- Visualize trends and correlations through interactive charts  
- Obtain predictions for crash types and severity  
- Explore key features and factors influencing the predictions  

## Conclusion  
This project demonstrates the potential of machine learning, distributed data processing with Apache Spark, and web-based applications to solve urban traffic challenges. By leveraging XGBoost for predictive accuracy, PySpark for scalability, and a web interface for accessibility, the study offers a comprehensive and practical framework for predictive analysis. These insights can aid in enhancing road safety, optimizing resource allocation, and supporting data-driven decision-making in urban planning and public safety initiatives.  
