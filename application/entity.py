import shap
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os
import pandas as pd
import datetime
import pdb

class Entity:
    # Initializing class member variables from dictionary obtained from user
    # This takes input a generalized dictionary of {feature_name:value} feature-value pairs 
    # and creates features as entity class's member variable
    def __init__(self, feature_dict):
        for key,value in feature_dict.items():
            setattr(self, key, value)

    # Get a list of names of all member variables or feature names of the class
    # Utility method that returns list of names of all feature which are entity class's member variable
    def get_feature_name_list(self):
        member_list = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        return member_list

    # Get a list of values of all member variables or feature names of the class
    # Utility method that returns list of values of all feature which are entity class's member variable
    def get_feature_val_list(self):
        member_list = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        member_val_list = [getattr(self,member_variable) for member_variable in member_list]
        return member_val_list

    # Get a dictionary of feature all features of the class
    # Utility method that returns a dictionary of {feature_name:value} feature-value pairs which are entity class's member variable
    def get_feature_dict(self):
        feat_dict = {}
        member_list = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        member_val_list = [getattr(self,member_variable) for member_variable in member_list]
        for idx in range(len(member_list)):
            feat_dict[member_list[idx]] = member_val_list[idx]
        return feat_dict

    # print all member variables of entity class
    # Utility method that prints all feature-value pairs which are entity class's member variable
    def print_vals(self):
        member_list = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        print("Values for customer are:")
        for member_variable in member_list:
            print(member_variable + ": "+ str(getattr(self,member_variable)))


    # scale all feature values in accordance to given scaler function
    # This method scales all feature variables according to scaling object and returns this as a list
    def scale_features(self,model_dict):
        # get the standard scaler object from trained model
        scaler = model_dict['scale_factor']
        orig_sample = np.array(self.get_feature_val_list()).reshape(1,-1)
        # scale the features entered by the user
        scaled_sample = scaler.transform(orig_sample)
        return scaled_sample

    # Utility method that prints all scaled feature-value pairs which are entity class's member variable
    def print_scaled_vals(self,model_dict):
        scaler = model_dict['scale_factor']
        orig_sample_names = self.get_feature_name_list()
        orig_sample_list = self.get_feature_val_list()
        orig_sample_np = np.array(orig_sample_list).reshape(1,-1)
        scaled_sample_np = scaler.transform(orig_sample_np)
        scaled_sample_list = scaled_sample_np.tolist()[0]
        #pdb.set_trace()
        for name, scaled_feature in zip(orig_sample_names, scaled_sample_list):
            print('Scaled ' + name + ": " + str(scaled_feature))


    # Return shap values of new sample
    # This method uses SHAP Explainer object which is already trained on training data to find SHAP values of user input
    def get_shap_values(self, explainer, model_dict):
        # This is the kernel shap explainer object already trained on training data
        shap_explainer =  explainer
        # Scale the user input features using scale_features method
        scaled_sample = self.scale_features(model_dict)
        # Use shap explainer object to find SHAP values of user input
        shap_value = shap_explainer.shap_values(scaled_sample)
        return shap_value

    # Return prediction of a new sample as tuple (prediction_class, probability)
    def get_prediction(self,model_dict):
        # get the scaled user input using scale_features method
        scaled_sample = self.scale_features(model_dict)
        # get the model object from the trained model. In our case it it trained Kernel SVM model object (SKLearn Object)
        clf = model_dict['model']
        pred = clf['model'].predict(np.array(scaled_sample).reshape(1, -1))[0]
        pred_proba = clf['model'].predict_proba(np.array(scaled_sample).reshape(1, -1))[0]
        #if pred == 1:
        #    print("This is a churner.")
        #else:
        #    print("This is a non-churner.")
        return pred, pred_proba

    # get a list of top 3 features in order of feature importance and dictionary containing their relative 
    # feature importance percentage derived from SHAP values 
    def get_local_feature_importance(self, explainer, model_dict):
        # get shap values for the user input
        shap_values = self.get_shap_values(explainer,model_dict)[1]
        # get all feature names which are member variables of entity class
        feat_names = self.get_feature_name_list()
        # take absolute values of all shap values
        shap_values_abs = np.argsort(np.abs(shap_values).mean(0))
        # calculate relative feature importance percentage using SHAP values
        shap_values_percentage = shap_values_abs/np.sum(shap_values_abs)*100
        # use sorting to find order list of all feature wrt feature importances
        ordered_indices_list = np.argsort(shap_values_abs).tolist()
        feat_names_sorted = [feat_names[i] for i in ordered_indices_list]
        shap_values_percentage_sorted = [shap_values_percentage[i] for i in ordered_indices_list]
        # Sorting fetures in decending order of feature importance
        feat_names_sorted.reverse()
        shap_values_percentage_sorted.reverse()
        # Creating dictionary of feature importance percentage
        feat_importance_dict = {k:v for k,v in zip(feat_names_sorted[:3],shap_values_percentage_sorted[:3])}
        return feat_names_sorted[:3], feat_importance_dict


# This is a global method that traines the SHAP on training data derived from trained model and then returns the trained Explainer object
def get_global_shap(model_dict):
    # get trained model object and training data from trained model
    clf = model_dict['model']
    X_train = model_dict['X_train']
    # use Kernel SHAP to explain test set predictions
    start_time = datetime.datetime.now()
    print("Shap is training on training data..........")
    # train shap explainer using trained model object and data
    explainer = shap.KernelExplainer(model=clf["model"].predict_proba, data=X_train, link="logit")
    end_time = datetime.datetime.now()
    print("Finished training shap on training data")
    exec_time = end_time - start_time
    print("Total execution time: ", exec_time)
    # Return the explainer object
    return explainer
    