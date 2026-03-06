This project applies machine learning techniques to predict real estate property prices based on property characteristics such as location, square footage, number of bedrooms, and other structural attributes. The system helps estimate property values by learning patterns from historical real estate transaction data.

The repository includes the dataset, a Jupyter notebook for data preprocessing, exploratory data analysis, and model development, along with a Flask web application that allows users to interactively predict property prices by entering property details.

Project Overview :
1.Real estate property data was cleaned and standardized to remove inconsistencies and missing values.
2.Exploratory Data Analysis was conducted to understand relationships between property features such as area, interior square footage, number of bedrooms, and sales price.
3.Feature engineering techniques were applied, including categorical encoding and derived features such as price per square foot.
4.Multiple regression models were developed and evaluated, including Linear Regression, Decision Tree Regression, Random Forest Regression, and LightGBM.
5.Hyperparameter tuning and cross validation were used to improve model performance and ensure generalization.
6.The final model was integrated into a Flask-based web application that enables users to input property attributes and receive real time price predictions.



Repository Structure :
1.Real_Estate_Dataset.csv - Dataset containing property attributes such as location, square footage, bedrooms, and sales prices.
2.Real_Estate_Prediction_Model.ipynb - Jupyter notebook containing data preprocessing, exploratory analysis, feature engineering, model training, and evaluation.
3.app.py - Flask web application that provides a user interface for predicting real estate prices based on user input.
4.scaler.pickle / trained model files - Saved model components used for prediction in the web application.



Technologies Used :
1.Python - Pandas, NumPy, Matplotlib, Seaborn
2.Machine Learning - Scikit-learn models including Linear Regression, Decision Tree, Random Forest
3.Gradient Boosting - LightGBM
4.Data Preprocessing and Feature Engineering
5.Flask for deployment and web interface



Model Architecture :
1.Data Cleaning and Standardization
2.Exploratory Data Analysis and Visualization
3.Feature Engineering and Categorical Encoding
4.Model Training with Multiple Regression Algorithms
5.Hyperparameter Optimization and Cross Validation
6.Model Evaluation using R² Score and Mean Squared Error
7.Deployment with Flask Web Application



Results :
Multiple machine learning models were evaluated to determine the most effective approach for predicting real estate prices.
Random Forest and LightGBM models achieved the strongest predictive performance, with LightGBM achieving an R² score of approximately 0.94, indicating strong predictive capability for estimating property values.
The trained model successfully captures relationships between property characteristics and sales price, enabling reliable price predictions through the Flask web interface.
