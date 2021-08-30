import numpy as np
from flask import Flask, request, jsonify, render_template
from joblib import load

from tensorflow.keras.models import load_model
app = Flask(__name__)

model = load_model("fruit_ann.h5")
sc=load("fruit.save")

@app.route('/')
def home():
    return render_template('fruit.html')

@app.route('/y_predict', methods=['POST'])
def y_predict():
    a = float(request.form['imass'])
    b = float(request.form['iwidth'])
    c = float(request.form['iheight'])
    d = request.form['fruit_label']
    if (d == "1"):
        d1,d2,d3,d4 = 1,0,0,0
    if (d == "2"):
        d1,d2,d3,d4 = 0,1,0,0
    if (d == "3"):
        d1,d2,d3,d4 = 0,0,1,0
    if (d == "4"):
        d1,d2,d3,d4 = 0,0,0,1
    z = request.form['fruit_subtype']
    if (z == "granny_smith"):
        o1,o2,o3,o4,o5,o6,o7,o8,o9,o10 = 1,0,0,0,0,0,0,0,0,0
    if (z == "mandarin"):
        o1,o2,o3,o4,o5,o6,o7,o8,o9,o10 = 0,1,0,0,0,0,0,0,0,0
    if (z == "braeburn"):
        o1,o2,o3,o4,o5,o6,o7,o8,o9,o10 = 0,0,1,0,0,0,0,0,0,0
    if (z == "golden_delicious"):
        o1,o2,o3,o4,o5,o6,o7,o8,o9,o10 = 0,0,0,1,0,0,0,0,0,0
    if (z == "cripps_pink"):
        o1,o2,o3,o4,o5,o6,o7,o8,o9,o10 = 0,0,0,0,1,0,0,0,0,0
    if (z == "spanish_jumbo"):
        o1,o2,o3,o4,o5,o6,o7,o8,o9,o10 = 0,0,0,0,0,1,0,0,0,0
    if (z == "selected_seconds"):
        o1,o2,o3,o4,o5,o6,o7,o8,o9,o10 = 0,0,0,0,0,0,1,0,0,0
    if (z == "turkey_navel"):
        o1,o2,o3,o4,o5,o6,o7,o8,o9,o10 = 0,0,0,0,0,0,0,1,0,0
    if (z == "spanish_belsan"):
        o1,o2,o3,o4,o5,o6,o7,o8,o9,o10 = 0,0,0,0,0,0,0,0,1,0
    if (z == "unknown"):
        o1,o2,o3,o4,o5,o6,o7,o8,o9,o10 = 0,0,0,0,0,0,0,0,0,1
    e = float(request.form['icolor'])
    total = [[a,b,c,d1,d2,d3,d4,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,e]]
    print(total)
    y_predict=model.predict(sc.transform(total))
    
    species = [ "apple","mandarin", "orange", "lemon"]
    prediction=species[np.argmax(y_predict)]
    print(prediction)

    if(prediction=='apple'):
        output = "The Fruit is Apple"
    
    elif(prediction=='mandarin'):
        output = "The fruit is Mandarin"
        
    elif(prediction=='orange'):
        output = "The fruit is Orange"
        
    elif(prediction=='lemon'):
        output = "The fruit is lemon"
        
    else:
        output = "Fruit not found"


    return render_template('fruit.html', prediction_text='Result: {}'.format(output))

if __name__ == "__main__":
    app.run(port=5000,debug=True)
    
