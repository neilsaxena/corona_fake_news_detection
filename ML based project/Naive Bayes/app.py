# Importing Libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Initializing App
classifier = pickle.load(open('naive_bayes_model.pkl', 'rb'))
cv=pickle.load(open('transform.pkl','rb'))
app = Flask(__name__)

# Our home page
@app.route('/')
def home():
    return render_template('home.html')


# Home page with prediction
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    user_input =  request.form["newsinput"]
    message = [user_input]
    
    message = cv.transform(message).toarray()
    
    pred = classifier.predict(message)
   
    result=''
    
    if pred[0] == 1:
        result = "It is not a fake News"
    else:
        result = "It is a Fake News"
        
    return render_template('home.html', prediction_text=result, msg = user_input)


# Starting Flask App
if __name__ == "__main__":
    app.run(debug=True)