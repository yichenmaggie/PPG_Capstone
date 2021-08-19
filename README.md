# PPG Capstone app

This repository contains a flask application that contains three components.

Through this application, the stakeholders will know - 
1. The prediction - whether a customer will churn or not? 
2. What are the most important features that contributed to such predictions
3. What changes are required to reverse the prediction - how to convert a churner to a non-churner? 

![App Pipeline](https://github.com/winnieshen96/ppg-capstone/blob/master/plots/App%20Pipeline.png)

The driver app is [main.py](https://github.com/winnieshen96/ppg-capstone/blob/master/main.py), and it uses code and templates in the [/application](https://github.com/winnieshen96/ppg-capstone/tree/master/application) folder.

## Flask Development

In the root directory, create the virtual environment. 

    python3 -m venv venv
  
After created the venv folder, use the following command to activate virtual environment.

    venv/Scripts/activate
  
After activating the virtual environment, use the [requirements.txt](https://github.com/winnieshen96/ppg-capstone/blob/master/requirements.txt) to install the required libraries.

    pip install -r requirements.txt
  
After installing the libraries, you can start the application by running the following command.

    flask run
  
and then go to the displayed localhost website to use the app.

## Application Structure

The front end html file is in [/templates](https://github.com/winnieshen96/ppg-capstone/tree/master/application/templates) folder.

The backend code has the three components mentioned above.

The classification model is in [model_training.py](https://github.com/winnieshen96/ppg-capstone/blob/master/application/model_training.py).

The feature importance component is in [entity.py](https://github.com/winnieshen96/ppg-capstone/blob/master/application/entity.py). The explainer is trained using the classification model and the entire dataset, and then it is used to calculate global and local importance. Global importance uses the  get_global_shap method, and local importance is object dependent, where an object is an entity and uses its own features to calculate the importance. 

The simulation algorithm we used to determine counterfactual is in [main_backend.py](https://github.com/winnieshen96/ppg-capstone/blob/master/application/main_backend.py).

## Modeling

The modeling attempts are in the [/notebooks](https://github.com/winnieshen96/ppg-capstone/tree/master/notebooks) folder.

We tried [Random Forest](https://github.com/winnieshen96/ppg-capstone/blob/master/notebooks/random_forest.ipynb), [SVM](https://github.com/winnieshen96/ppg-capstone/blob/master/notebooks/svm.ipynb), [Decision Trees](https://github.com/winnieshen96/ppg-capstone/blob/master/notebooks/PPG%20-%20Logistic%20Regression.ipynb), [Logistic Regression](https://github.com/winnieshen96/ppg-capstone/blob/master/notebooks/PPG%20-%20Logistic%20Regression.ipynb)

The current best model is SVM, and the evaluation metrics are as follows.

![Evaluation](https://github.com/winnieshen96/ppg-capstone/blob/master/plots/Evaluation.png)

### Data Exploration

The data is the aggregate form of a customer churn dataset in [/data/ppg](https://github.com/winnieshen96/ppg-capstone/tree/master/data/ppg).

The data exploration plots are in the [/plots](https://github.com/winnieshen96/ppg-capstone/tree/master/plots) folder.

### What-if Tool

We tried the Google What-if Tool to explore counterfactuals. The notebook is in [svm_colab_wit.ipynb](https://github.com/winnieshen96/ppg-capstone/blob/master/notebooks/svm_colab_wit.ipynb)

