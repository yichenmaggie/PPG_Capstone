import shap
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os
import pandas as pd
import datetime
from model_training import *

class Customer:
    def __init__(self, invoice_count, total_product, avg_pruchase_freq, avg_spend_ttm):
        self.invoice_count = invoice_count      
        self.total_product = total_product
        self.avg_pruchase_freq = avg_pruchase_freq
        self.avg_spend_ttm = avg_spend_ttm
        self.scaled_invoice_count  = 0
        self.scaled_total_product = 0
        self.scaled_avg_pruchase_freq = 0
        self.scaled_avg_spend_ttm = 0

    def print_orig_vals(self):
        print("Values for customer are:")
        print("Invoice count: %d \nTotal product: %d \nAvg Purchase Freq: %.2f \nAvg spend ttm: %.2f"%(self.invoice_count, self.total_product,\
            self.avg_pruchase_freq, self.avg_spend_ttm))

    def print_scaled_vals(self):
        print("Scaled Values for customer are:")
        print("Invoice count: %.5f \nTotal product: %.5f \nAvg Purchase Freq: %.5f \nAvg spend ttm: %.5f"%(self.scaled_invoice_count, self.scaled_total_product,\
            self.scaled_avg_pruchase_freq, self.scaled_avg_spend_ttm))

    def scale_features(self,scaler):
        orig_sample = np.array([self.invoice_count, self.total_product, self.avg_pruchase_freq, self.avg_spend_ttm]).reshape(1,-1)
        scaled_sample = scaler.transform(orig_sample)
        self.scaled_invoice_count  = scaled_sample[0][0]
        self.scaled_total_product = scaled_sample[0][1]
        self.scaled_avg_pruchase_freq = scaled_sample[0][2]
        self.scaled_avg_spend_ttm = scaled_sample[0][3]
        return scaled_sample

    def get_shap_values(self,explainer,scaler):
        shap_explainer =  explainer
        scaled_sample = self.scale_features(scaler)
        shap_value = shap_explainer.shap_values(scaled_sample)
        return shap_value

    def get_prediction(self,model_dict):
        scale_sample = np.array([self.scaled_invoice_count, self.scaled_total_product, self.scaled_avg_pruchase_freq,
                                 self.scaled_avg_spend_ttm])
        pred = model_dict['model'].predict(np.array(scale_sample).reshape(1, -1))[0]
        if pred == 1:
            print("This is a churner.")
        else:
            print("This is a non-churner.")

def get_global_shap(model_dict):
    svm_clf = model_dict['model']
    X_train = model_dict['X_train']
    #y_train = load_dict['y_train']
    #X_test = load_dict['x_test']
    #y_test = load_dict['y_test']
    #X_test_orig = load_dict['x_test_orig']
    # use Kernel SHAP to explain test set predictions
    start_time = datetime.datetime.now()
    print("Shap is training on training data..........")
    explainer = shap.KernelExplainer(model=svm_clf["model"].predict_proba, data=X_train, link="logit")
    end_time = datetime.datetime.now()
    print("Finished training shap on training data")
    exec_time = end_time - start_time
    print("Total execution time: ", exec_time)
    return explainer

#Testing
# from model_training import *
#print("Here")
#model_dict = train_model()
#print("Model trained")
#c1 = Customer(1,666,0.000000,31.062)
#c1.print_orig_vals()
#print("Scaling features")
#c1.scale_features(model_dict['scale_factor'])
#c1.print_scaled_vals()
#print("Features scaled")
#explainer = get_global_shap(model_dict)
#lst = c1.get_shap_values(explainer,model_dict['scale_factor'])
#print(lst)