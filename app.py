
#Step 1: Import libraries
import numpy as np
from flask import Flask, request, render_template
import pickle

#Step 2: Building the app
my_app = Flask(__name__)

#Step 3: Loading the pickle file and read it
regression_model = pickle.load(open('reg_model.pkl', 'rb'))

#Step 4: Set the HOME-PAGE of the web-app
@my_app.route('/')
def home():
    return render_template('index.html')

#Step 5: Set the output page
@my_app.route('/prediction',methods=['POST'])
def predict():
    
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = regression_model.predict(final_features)

    outcome = round(prediction[0], 2)

    return render_template('index.html', prediction_text='The annual rental income should be ${}'.format(outcome))

#Run the FLASK app
if __name__ == "__main__":
    my_app.run(debug=False)
