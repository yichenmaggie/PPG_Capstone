#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# ## Read in data and pass to the backend

# In[2]:


# This function take a csv file with only one row
# and return a dictionary called features (key(feature name): value(value))
def passToBackEnd(file):
    data=pd.read_csv(file)
    feature_name=data.columns
    values=data.loc[0]
    
    features={}
    for i in range(len(feature_name)):
        features[feature_name[i]]=values[i]
    
    return features


# In[3]:


#test
test=passToBackEnd("test.csv")
test


# ## Get result back from the back end

# In[4]:


# This function takes a customer_object passed from the backend.
# Return a result string that would display at the front-end interface.

def front_end_display(customer_object):
    features= getattr(customer_object, "features")
    old_predictions=getattr(customer_object, "old_predictions")
#     shap_values=getattr(customer_object, "shap_values")
#     most_important_features=getattr(customer_object, "most_important_features")
#     new_features=getattr(customer_object, "new_features")
    tweaked_feature=getattr(customer_object, "tweaked_feature")
    new_predictions=getattr(customer_object, "new_predictions")
    
    classification=""
    
    if  old_predictions["churn"]>old_predictions["non-churn"]:
        classification="churn"
    else:
        classification="non-churn"
    
    tweaked_feature_name=list(tweaked_feature.keys())[0]
    tweaked_feature_value=tweaked_feature[tweaked_feature_name]
    
    result=""
    
    if classification=="churn":
        result+="This customer is a \"churner\". \n"
        result+="For this customer, we suggest that we should tweak {0} from {1} to {2}, because that is the most important feature in this prediction. \n".format(tweaked_feature_name, features[tweaked_feature_name], tweaked_feature_value)
        result+="After applying the change, we can see the probability of {0} decreases to {1} and the probability of {2} increases to {3} \n".format("churn", new_predictions["churn"] , "non-churn", new_predictions["non-churn"])
    else:
        result+="This customer is a \"non-churner\". \n"
        result+="For this customer, we warn that if {0} change from {1} to {2}. \n".format(tweaked_feature_name, features[tweaked_feature_name], tweaked_feature_value)
        result+="The probability of {0} decreases to {1}, and the probability of {2} increases to {3}, which is something to avoid.".format("non-churn", new_predictions["non-churn"],"churn", new_predictions["churn"])
        

    return result
        


# In[5]:


#test
class Test:
    features={"invoice_count": "10"}
    old_predictions={"churn": 0.4, "non-churn":0.6}
    tweaked_feature={"invoice_count": "4"}
    new_predictions={"churn": 0.7, "non-churn":0.3}


# In[6]:


t1=Test()


# In[7]:


print(front_end_display(t1))


# In[8]:


t2=Test()
t2.features={"invoice_count": "4"}
t2.old_predictions={"churn": 0.6, "non-churn":0.4}
t2.tweaked_feature={"invoice_count": "10"}
t2.new_predictions={"churn": 0.3, "non-churn":0.7}


# In[9]:


print(front_end_display(t2))

