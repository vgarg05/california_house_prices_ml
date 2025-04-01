from flask import Flask,jsonify,render_template,request
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

application=Flask(__name__)
app=application

# import ridge regressor and standard scalar pickle
model=pickle.load(open("models/regressor.pkl","rb"))
standard_scaler=pickle.load(open("models/scaler.pkl","rb"))

@app.route("/") 
def index():
    return render_template("index.html")

@app.route("/predictdata",methods=["GET","POST"])
def predict_datapoint():
    if request.method=="POST":
        MedInc=float(request.form.get('MedInc'))
        HouseAge = float(request.form.get('HouseAge'))
        AveRooms = float(request.form.get('AveRooms'))
        AveBedrms = float(request.form.get('AveBedrms')) 
        Population = float(request.form.get('Population')) 
        AveOccup = float(request.form.get('AveOccup'))
        Latitude = float(request.form.get('Latitude')) 
        Longitude = float(request.form.get('Longitude')) 
        data=standard_scaler.transform([[MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude]])
        result=model.predict(data)
        return render_template("home.html",results=result)
    else:
        return render_template("home.html")


if __name__=="__main__":
    app.run(debug=True)
