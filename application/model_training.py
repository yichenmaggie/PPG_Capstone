import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE #Over sampling


# Load general utilities
# ----------------------
import pandas as pd
import datetime
import numpy as np
import pickle
import time

# Load sklearn utilities
# ----------------------
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, brier_score_loss, mean_squared_error, r2_score

from sklearn.calibration import calibration_curve

# Other Packages
# -------------- 
from sklearn.tree import export_graphviz
# from scipy.interpolate import spline

# Load debugger, if required
#import pixiedust
pd.options.mode.chained_assignment = None #'warn'

# suppress all warnings
import warnings
warnings.filterwarnings("ignore")


def train_model():
    df = pd.read_csv('data/ppg/CMU_resample.csv')
    # Set the column in the dataset you wish for the model to predict
    # label_column = 'label'
    
    # Make the label column numeric (0 and 1), for use in our model.
    # In this case, examples with a target value of '>50K' are considered to be in
    # the '1' (positive) class and all other examples are considered to be in the
    # '0' (negative) class.
    # make_label_column_numeric(df, label_column, lambda val: val == 1)

    # Set list of all columns from the dataset we will use for model input.
    input_features = ["invoice_count","total_products","avg_purchase_frequency","avg_spend_ttm"]

    # Create a list containing all input features and the label column
    #features_and_labels = input_features + [label_column]

    X = df[input_features] # Features
    y = df.label # Target variable

    # Split dataset into training set and test set
    default_seed = 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # 80% training and 20% test

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    #X_non_transformed = X_test.to_numpy()
    X_test = scaler.transform(X_test)
    
    ## Instantiate smote and balance training data only
    oversample = SMOTE()
    X_train, y_train = oversample.fit_resample(X_train, y_train)

    ## Compute and print percentage of high quality wine after balancing
    print('Percentage of high quality counts in the balanced data:{}%'.format(np.sum(y_train==1)/len(y_train)*100))
    
    data_dict = {"X_train":X_train, "X_test":X_test, "y_train":y_train, "y_test":y_test}
    default_seed = 1

    cv_parameters = {'kernel':['rbf'], 'class_weight':["balanced"]}
    svm_clf = SVC(probability=True)
    # svm_clf = fit_classification(svm_clf, data_dict,
    #                           cv_parameters = cv_parameters,
    #                           scoring = 'precision',
    #                           model_name = "Balanced SVM",
    #                           random_state = default_seed,
    #                           output_to_file = False,
    #                           print_to_screen = False)
    svm_clf = fit_classification(model=svm_clf, data_dict=data_dict,
                                 cv_parameters=cv_parameters, 
                                 random_state = default_seed)
    return_dict = {"X_train":X_train, "X_test":X_test, "y_train":y_train, "y_test":y_test, "model": svm_clf, "scale_factor": scaler}
    
    with open('model_info.pickle', 'wb') as handle:
        pickle.dump(return_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)



def fit_classification(model, data_dict,
                       cv_parameters={},
                       scoring='recall',
                       random_state=1):

    np.random.seed(random_state)
    #   Step 1 - Load the data
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']

    X_test = data_dict['X_test']
    # y_test = data_dict['y_test']

    #   Step 2 - Fit the model
    cv_model = GridSearchCV(model, cv_parameters, scoring=scoring)
    cv_model.fit(X_train, y_train)
    best_model = cv_model.best_estimator_

    #   Step 3 - Evaluate the model
    # If possible, make probability predictions
    try:
        y_pred_probs = best_model.predict_proba(X_test)[:, 1]
        probs_predicted = True
    except:
        probs_predicted = False

    # Make predictions; if we were able to find probabilities, use
    # the threshold that maximizes the accuracy in the training set.
    # If not, just use the learner's predict function
    if probs_predicted:
        y_train_pred_probs = best_model.predict_proba(X_train)[:, 1]
        fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_train_pred_probs)

        true_pos_train = tpr_train * (y_train.sum())
        true_neg_train = (1 - fpr_train) * (1 - y_train).sum()

        best_threshold_index = np.argmax(true_pos_train + true_neg_train)
        best_threshold = 1 if best_threshold_index == 0 else thresholds_train[best_threshold_index]

        y_pred = (y_pred_probs > best_threshold)
    else:
        y_pred = best_model.predict(X_test)

    # Return the model predictions, and the
    # test set
    # -------------------------------------
    out = {'model': best_model, 'y_pred_labels': y_pred}

    if probs_predicted:
        out.update({'y_pred_probs': y_pred_probs})
        out.update({'pred_proba': model.predict_proba})
    else:
        y_pred_score = best_model.decision_function(X_test)
        out.update({'y_pred_score': y_pred_score})
    return out


if __name__== "__main__":
    train_model()