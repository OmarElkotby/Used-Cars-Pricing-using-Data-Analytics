
from flask import Flask, render_template, request
from OMhelpers.encoders import *
import numpy as np
import joblib


OMapp=Flask(__name__,template_folder="OMTemplates")

model = joblib.load('C:/Users/Pc/OneDrive/Desktop/2nd graduation project/project dataset_notebook/model.om',"readwrite")
scaler = joblib.load('C:/Users/Pc/OneDrive/Desktop/2nd graduation project/project dataset_notebook/scaler.om',"readwrite")

@OMapp.route('/')
def home():
    return render_template('OMpage.html')

@OMapp.route('/Indian_Used_Cars_Price_Prediction', methods=['POST'])
def Indian_Used_Cars_Price_Prediction():
    
    year = request.form["year"]
    kilometers= request.form["kilometers_driven"]
    mileage= request.form["mileage"]
    engine = request.form["engine"]
    power = request.form['power']
    seats  = request.form["seats"]
    name = carName_encoders[request.form["NameOfCar"]]
    location =Location_encoders[request.form["LocationOfCar"]]
    fuel_type = Fuel_encoders[request.form["FuelTypeOfCar"]]
    transmission = Transmission_encoders[request.form['TransmissionOfCar']]
    owner_type = OwnerType_encoders[request.form['OwnerType']]
    x=np.array([year,kilometers,mileage,engine,power,seats,name,location,fuel_type,transmission,owner_type])
    x2=scaler.transform([x])
    
    carprice=model.predict(x2)
    
    return render_template('OMpage.html',price_text= "Price of car is : {}$".format(carprice))
    
if __name__ == "__main__":
    
    
    OMapp.debug=True
    OMapp.run()