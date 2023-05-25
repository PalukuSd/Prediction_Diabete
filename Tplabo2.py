from distutils.log import debug
from flask import Flask,request,jsonify,render_template,redirect,url_for
import sklearn
import pickle
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

app=Flask(__name__)
models=joblib.load(open('ExamenLabo3.ml','rb'))
dict_class_lesion={
    0:"Negatif",
    1:"Positif"
}

@app.route('/')

def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])

def predict():
    import joblib
    models=joblib.load('ExamenLabo3.ml')
    int_futures=[float(i) for i in request.form.values()]
    dernier_futures=[np.array(int_futures)]
    dernier_futures=np.array([dernier_futures]).reshape(1,8)
    predire=models.predict(dernier_futures)

    # pred_class=predire.argmax(axis=-1)
    # prediction=dict_class_lesion[predire[0]]
    # result=str(prediction)

    if(models.predict(dernier_futures)==1):
        predire="Negatif"
    else:
        predire="Positif"

    return render_template('index.html', prediction_text_="Votre Test est : {}".format(predire))
if __name__=='__main__':
    app.run(debug=True)