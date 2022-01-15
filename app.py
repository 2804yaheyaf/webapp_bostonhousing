from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model_file.pkl','rb'))


@app.route("/")
def hello():
    return render_template("index.html")


@app.route("/sub", methods=['POST'])
def submit():
    if request.method == "POST":
        name=request.form["username"]
    
    return render_template("sub.html", n=name)


# to use predict button in web app
@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)


    output = round(prediction[0], 2) 
    
    
    return render_template('index.html', prediction_text='Predicted price of the House is ${}'.format(output))


@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__== "__main__":
    app.run(debug=True)
