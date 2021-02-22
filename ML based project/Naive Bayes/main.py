# Importing Libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import numpy as np
import itertools
import pickle

# Importing dataset
df=pd.read_csv('corona_data.csv')

# Get the Independent Features

X=df.drop('label',axis=1)

# Get the Dependent features
y=df['label']


# Preprocessing the data
df=df.dropna()

messages=df.copy()

messages.reset_index(inplace=True)


ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['news'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
    
cv = CountVectorizer(max_features=5000,ngram_range=(1,3))
X = cv.fit_transform(corpus).toarray()

y=messages['label']

# Training model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=0)

classifier=MultinomialNB()
classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)

score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)

cm = metrics.confusion_matrix(y_test, pred)


#saving model and countvectoriser

pickle.dump(classifier, open('naive_bayes_model.pkl','wb'))
pickle.dump(cv,open('transform.pkl','wb'))
