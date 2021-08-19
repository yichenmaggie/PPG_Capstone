from flask import Flask, render_template, request, json, session, redirect, url_for
import os
import csv
from werkzeug.utils import secure_filename
import io

app = Flask(__name__)

# Create a directory in a known location to save files to.
UPLOAD_FOLDER = 'upload'
basedir = os.path.abspath(os.path.dirname(__file__))
uploads_dir = os.path.join(basedir, UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods = ['POST'])
def upload():
    if request.method == 'POST':

        # Create a dictionary for the uploaded features
        f = request.files['fileupload']
        stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
        csv_input = csv.reader(stream, delimiter=',')
        features = {}
        headers = []
        values = []
        for index, row in enumerate(csv_input):
            if index == 0:
                headers = row
            if index == 1:
                values = row
        for i in range(len(headers)):
            features[headers[i]] = float(values[i])
        print(features)
        
        # create a dictionary for the label names
        labels = {0:request.form['name_0'], 1:request.form['name_1']}
        print(labels)

        # create a variable for number of steps
        num_of_steps = int(request.form['num_of_steps'])
        print(num_of_steps)

        # call the back end function and pass in features, labels, and num_of_steps

        from application.main_backend import start_process

        prediction_dic=start_process(features, num_of_steps)

        print(prediction_dic)
        #{dict of original features : values}
        features= prediction_dic["feature"]
        # 0 or 1
        old_prediction_label=prediction_dic["old_prediction_label"]
        # eg: {“1”: 0.4, “0”: 0.6}
        old_prediction_prob=prediction_dic['old_prediction_probability']
        # 0 or 1
        new_prediction_label=prediction_dic["new_prediction_label"]
        # eg: {“1”: 0.7, “0”: 0.3}
        new_prediction_prob=prediction_dic['new_prediction_probability']
        # {feature_name: new_value}, at most two features and at least one feature
        tweaked_feature=prediction_dic['tweak_feature']
        # [list of top3 important features]
        most_important_features=prediction_dic['most_important_features']

        for f in most_important_features:
            most_important_features[f]=round(most_important_features[f],2)
            
        tweaked_feature_names=list(tweaked_feature.keys())
        tweaked_feature_name1=tweaked_feature_names[0]
        tweaked_feature_name2=""
        if len(tweaked_feature_names)>1:
            tweaked_feature_name2=tweaked_feature_names[1]
            
        result=""
            
        if old_prediction_label==new_prediction_label:
            result="There is no better optimization can be found."
        else:
            if tweaked_feature_name2=="":
                result+="This customer is a "+labels[old_prediction_label]+". "
                result+= "For this customer, if {0} changes from {1} to {2} and other features are kept the same, this customer would become a {3}. ".format(tweaked_feature_name1,
                                                                                                                    features[tweaked_feature_name1],
                                                                                                                    round(tweaked_feature[tweaked_feature_name1],2),
                                                                                                                    labels[new_prediction_label])
                result+= "If {0} is a good sign, please be careful not to make that change happen; if not, try tweaking {1} to our suggestion.".format(labels[old_prediction_label],
                                                                                                                                    tweaked_feature_name1)
            else:
                result+="This customer is a "+labels[old_prediction_label]+"."
                result+= "For this customer, if {0} changes from {1} to {2} and {3} change from {4} to {5} and other features are kept the same, this customer would become a {6}. ".format(tweaked_feature_name1,
                                                                                                                    features[tweaked_feature_name1],
                                                                                                                    round(tweaked_feature[tweaked_feature_name1],2),
                                                                                                                    tweaked_feature_name2,
                                                                                                                    features[tweaked_feature_name2],
                                                                                                                    round(tweaked_feature[tweaked_feature_name2],2),
                                                                                                                    labels[new_prediction_label])
                result+= "If {0} is a good sign, please be be careful not to make that change happen; if not, try tweaking {1} and {2} to our suggestion.".format(labels[old_prediction_label],
                                                                                                                            tweaked_feature_name1, tweaked_feature_name2)
        
        # display it to the most important features


    return render_template("index.html", result = result, features = most_important_features)
