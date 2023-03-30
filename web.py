from flask import Flask,render_template,request
import pickle
import numpy as np

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')
    return redirect(url_for("predict"))
@app.route('/predict',methods =['POST'])
def predict():
    iris=[float(x)for x in request.form.values()]
    final_iris= [np.array(iris)]
    output=model.predict(final_iris)
    return render_template('res.html',prediction_text="The flower classification is.{}".format(output))

if __name__ =='__main__':
    app.run(debug=True)