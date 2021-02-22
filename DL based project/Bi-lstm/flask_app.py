# Importing Libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initializing App
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
    message = user_input
        
    ps = PorterStemmer()

    user_input = re.sub('[^a-zA-Z]', ' ', user_input)
    user_input = user_input.lower()
    user_input = user_input.split()
    user_input = [ps.stem(word) for word in user_input if not word in stopwords.words('english')]
    user_input = ' '.join(user_input)

    voc_size=5000       # Vocabulary size
    onehot_user_input=[one_hot(user_input,voc_size)]

    sent_length=20
    embedded_user_input=pad_sequences(onehot_user_input,padding='pre',maxlen=sent_length)
    embedded_user_input = np.array(embedded_user_input)

    y_prediction=loaded_model.predict_classes(embedded_user_input)
        
    result=''
    
    if y_prediction[0][0] == 1:
        result = "It is not a fake News"
    else:
        result = "It is a Fake News"
        
    return render_template('home.html', prediction_text=result, msg = message)


# Starting Flask App
if __name__ == "__main__":
    
    # Loading Bidirectional LSTM model
    loaded_model = load_model('Bi_lstm_model.h5')
    app.run(debug=True)