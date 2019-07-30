from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

app=Flask(__name__)
Bootstrap(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    df=pd.read_csv('https://raw.githubusercontent.com/Jcharis/Machine-Learning-Web-Apps/master/Gender-Classifier-ML-App-with-Flask%20%2B%20Bootstrap%20/data/names_dataset.csv')
    df_X=df.name
    df_Y=df.sex
    df_name=df
    df_name.sex.replace({'F':0,'M':1},inplace=True)
    Xfeature=df_name['name']
    cv=CountVectorizer()
    X=cv.fit_transform(Xfeature)
    feature=X
    label=df_name.sex
    X_train,X_test,y_train,y_test=train_test_split(feature,label,test_size=0.33,random_state=42)
    clf = MultinomialNB()
    train=clf.fit(X_train,y_train)

    if request.method=='POST':
        namequery=request.form['namequery']
        data=[namequery]
        vect=cv.transform(data).toarray()
        my_prediction=clf.predict(vect)
    return render_template('result.html',prediction=my_prediction,name=namequery.upper())
if __name__=='__main__':
    app.run(debug=True)
