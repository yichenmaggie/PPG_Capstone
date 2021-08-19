# all imports
from application.model_training import train_model
from application.entity import Entity,get_global_shap
import pickle

# this function is the simulation process
def simulation(feature, entity, model_dict, numOfSteps=15000):
    # initialize step number
    step = 0
    # get the original prediction
    original_label, _ = entity.get_prediction(model_dict)
    # get the original feature values
    _dict = entity.get_feature_dict()
    increase = _dict[feature]
    decrease = _dict[feature]
    # change value for each step
    change_val = _dict[feature]*0.05
    
    # loop to simulate
    while True:
        # increase the feature value by 5%
        increase += change_val
        _dict[feature] = increase
        entity_increase = Entity(_dict)
        # get the label after changes
        label_increase, label_increase_proba = entity_increase.get_prediction(model_dict)
        
        # decrease the feature value by 5%
        decrease -= change_val
        _dict[feature] = decrease
        entity_decrease = Entity(_dict)
        #get the label after changes
        label_decrease, label_decrease_proba = entity_decrease.get_prediction(model_dict)
        
        # if label is reversed, print the result and return the details
        if label_decrease != original_label:
            print("Decreased " + str(step) + " steps, label reversed.")
            print("Changed " + feature + " to " + str(decrease))
            return step,label_decrease,label_decrease_proba, entity_decrease
        elif label_increase != original_label:
            print("Increased " + str(step) + " steps, label reversed.")
            print("Changed " + feature + " to " + str(increase))
            return step, label_increase, label_increase_proba, entity_increase
        # if the label is not reversed given the certain step number, print the result and return the details
        if step == numOfSteps:
            print("Explored " + str(numOfSteps) + " steps, label not reversed.")
            return step, label_increase, label_increase_proba, entity_increase
        # increase step number by 1
        step += 1

# this function is the simulation process in pair
def simulation_pair(feature1, feature2, entity, model_dict, numOfSteps=10000):
    # initialize step number
    step = 0
    # get the original prediction
    original_label, _ = entity.get_prediction(model_dict)
    # get the original two feature values
    _dict = entity.get_feature_dict()
    increase_feat1 = _dict[feature1]
    decrease_feat1 = _dict[feature1]
    increase_feat2 = _dict[feature2]
    decrease_feat2 = _dict[feature2] 
    # change value for each step for two features
    change_val1 = _dict[feature1]*0.05
    change_val2 = _dict[feature2]*0.05
    
    # loop to simulate
    while True:
        # increase the feature 1 value by 5%
        increase_feat1 += change_val1
        # decrease the feature 1 value by 5%
        decrease_feat1 -= change_val1
        # increase the feature 2 value by 5%
        increase_feat2 += change_val2
        # decrease the feature 2 value by 5%
        decrease_feat2 -= change_val2

        ## feature 1 increase feature 2 increase
        _dict[feature1] = increase_feat1
        _dict[feature2] = increase_feat2
        entity_ii = Entity(_dict)
        #get the label after changes
        label_ii, label_ii_proba = entity_ii.get_prediction(model_dict)

        ## feature 1 increase feature 2 decrease
        _dict[feature1] = increase_feat1
        _dict[feature2] = decrease_feat2
        entity_id = Entity(_dict)
        #get the label after changes
        label_id, label_id_proba = entity_id.get_prediction(model_dict)

        ## feature 1 decrease feature 2 increase
        _dict[feature1] = decrease_feat1
        _dict[feature2] = increase_feat2
        entity_di = Entity(_dict)
        #get the label after changes
        label_di, label_di_proba = entity_di.get_prediction(model_dict)

        ## feature 1 decrease feature 2 decrease
        _dict[feature1] = decrease_feat1
        _dict[feature2] = decrease_feat2
        entity_dd = Entity(_dict)
        #get the label after changes
        label_dd, label_dd_proba = entity_dd.get_prediction(model_dict)

        # if label is reversed, return the details
        if label_ii != original_label:
            return step,label_ii,label_ii_proba, entity_ii
        elif label_id != original_label:
            return step, label_id, label_id_proba, entity_id
        elif label_di != original_label:
            return step, label_di, label_di_proba, entity_di
        elif label_dd != original_label:
            return step, label_dd, label_dd_proba, entity_dd
        # if the label is not reversed given the certain step number, print the result and return the details
        if step == numOfSteps:
            print("Explored " + str(numOfSteps) + " steps, label not reversed.")
            return step, label_ii, label_ii_proba, entity_ii
        # increase step number by 1
        step += 1

# this function is to start the whole simulation process and return the dictionary
def start_process(feature_dict,noOfSteps = 5000):
    # get model info
    print("Getting model info")
    # read the pickle file and get model_dict
    with open('model_info.pickle', 'rb') as handle:
        model_dict = pickle.load(handle)
    # get the global importance of shap value
    print("Fitting SHAP explainer on the training Data")
    explainer_object = get_global_shap(model_dict)
    # create entity object
    print("Creating object for new data")
    entity_object = Entity(feature_dict)
    # get the current prediction and predict probability
    print("Getting current prediction of the new Data")
    curr_pred, curr_pred_proba = entity_object.get_prediction(model_dict)
    # get the local feature importances
    print("Getting local feature importances")
    feat_importance_lst, feat_importance_dict = entity_object.get_local_feature_importance(explainer_object, model_dict)
    
    # initialize the dictionary to return
    return_dict = {}
    # put feature values, original predicton, original probabiliy and the three most important features
    return_dict['feature'] = feature_dict
    return_dict['old_prediction_label'] = curr_pred
    return_dict['old_prediction_probability'] = curr_pred_proba
    return_dict['most_important_features'] = feat_importance_dict

    # run simulation to get tweak feature which changes prediction
    # simulation wrt most important feature
    n_simulation = noOfSteps
    step1, label1, label_proba1, entity1 = simulation(feat_importance_lst[0], entity_object, model_dict, n_simulation)
    # simulation wrt second most important feature
    step2, label2, label_proba2, entity2 = simulation(feat_importance_lst[1], entity_object, model_dict, n_simulation)

    #get the feature with the minimum step size to reverse the label and put in the dict
    if step1 < step2:
        print(feat_importance_lst[0] + " could reverse the label with least steps of " + str(step1))
        return_dict['new_prediction_label'] = label1
        return_dict['new_prediction_probability'] = label_proba1
        tweak_dict = {}
        tweak_dict[feat_importance_lst[0]] = entity1.get_feature_dict()[feat_importance_lst[0]]
        return_dict['tweak_feature'] = tweak_dict

    elif step1 > step2:
        print(feat_importance_lst[1] + " could reverse the label with least steps of " + str(step2))
        return_dict['new_prediction_label'] = label2
        return_dict['new_prediction_probability'] = label_proba2
        tweak_dict = {}
        tweak_dict[feat_importance_lst[1]] = entity2.get_feature_dict()[feat_importance_lst[1]]
        return_dict['tweak_feature'] = tweak_dict

    elif (step1 == step2) and (step1 < n_simulation):
        print(feat_importance_lst[0] + " and " + feat_importance_lst[1] + 
        " both can reverse the label with " + str(step1) + " steps.")
        return_dict['new_prediction_label'] = label1
        return_dict['new_prediction_probability'] = label_proba1
        tweak_dict = {}
        tweak_dict[feat_importance_lst[0]] = entity1.get_feature_dict()[feat_importance_lst[0]]
        return_dict['tweak_feature'] = tweak_dict
    # if cannot reverse the label, do the simulation in pair and put value in the dict
    else:
        print(feat_importance_lst[0] + " and " + feat_importance_lst[1] + 
        " both can not individually reverse the label within " + str(step1) + " steps.")
        print("Now trying simulation in pair............")
        n_pair_simulation = noOfSteps
        _, label_pair, label_pair_proba, entity_pair = simulation_pair(feat_importance_lst[0], feat_importance_lst[1], entity_object, model_dict, n_pair_simulation)

        return_dict['new_prediction_label'] = label_pair
        return_dict['new_prediction_probability'] = label_pair_proba
        tweak_dict = {}
        tweak_dict[feat_importance_lst[0]] = entity_pair.get_feature_dict()[feat_importance_lst[0]]
        tweak_dict[feat_importance_lst[1]] = entity_pair.get_feature_dict()[feat_importance_lst[1]]
        return_dict['tweak_feature'] = tweak_dict
    # return the dictionary to front end
    return return_dict
