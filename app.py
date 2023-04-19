from flask import Flask, render_template, request,send_file
import pickle
import spacy
import pandas as pd


#loading the english language small model of spacy
en = spacy.load('en_core_web_sm')
sw_spacy = en.Defaults.stop_words

filename = 'finalized_model.pkl'
classifier = pickle.load(open(filename, 'rb'))
tf = pickle.load(open('tf-transform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        stw=(" ".join([word for word in message.split() if word not in (sw_spacy)]))
        data = [stw]
        print(data)
        vect = tf.transform(data).toarray()
        my_prediction = classifier.predict(vect)
        return render_template('result.html', prediction=my_prediction)


@app.route('/predict2',methods=['POST'])
def predict2():
    if request.method == 'POST':
        uploaded_file=request.files['file']
        
        if not uploaded_file:
            return render_template('home.html',error_message='Please upload a file')
        
        df=pd.read_csv(uploaded_file,encoding="ISO-8859-1")
        df['clean']=df['v2'].apply(lambda x:''.join ([word for word in x.split() if word not in (sw_spacy)]))
        
        vect = tf.transform(df['clean']).toarray()
        my_prediction = classifier.predict(vect)
        
        a=[]
        for i in range(len(my_prediction)):
            if my_prediction[i]==0:
                a.append("ham")
            else:
                a.append("spam")
        
        df['Predicted']=a
        
    
        
        
        df.to_csv('Prediction.csv',index=False)
        return send_file('Prediction.csv',as_attachment=True)
    
    return render_template('home.html')

if __name__ == '__main__':
	app.run(debug=True)