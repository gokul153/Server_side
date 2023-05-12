import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from flask import Flask, request, jsonify
import lightgbm as lgb
#import tensorflow
#from tensorflow.keras.preprocessing import image
import numpy as np
import requests, json
"""import cv2
import base64
import io
from PIL import Image
import base64
from io import BytesIO
import urllib.parse"""
#firebase = Firebase(config)

#disease_categories= ["Bacterial Spot","Early Blight","Late Blight","Leaf Mold","Septoria Leaf Spot","Spider Mites","Target Spot","Yellow Leaf Curl Virus","Mosaic Virus","Healthy"]
crop_categories = ["Rice","Maize","Chickpea","Kidneybeans","Pigeonpeas","Mothbeans","Mungbean","Blackgram","Lentil","Pomegranate","Banana","Mango","Grapes","Watermelon","Muskmelon","Apple","Orange","Papaya","Coconut","Cotton","Jute","coffee"]
bst = lgb.Booster(model_file='crop_reccomendation_model.txt') 
model = tensorflow.keras.models.load_model("leaf_disease_model.h5")
application = Flask(__name__)
app = application
app.config["DEBUG"]=True


@app.route("/")
def hello_world():
    return "Models loaded"

@app.route("/crop",methods = ['GET'])
def crop_rec():
    n= request.args.get('n')
    p= request.args.get('p')
    k= request.args.get('k')
    ph= request.args.get('ph')
    r= request.args.get('r')
    api_key = "8f9f9ea0bb743b5ee9a0e2ec0f54e031"
    city_name = "Thiruvananthapuram"
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()
    if x["cod"] != "404":
        y = x["main"]

        temperature = y["temp"]
        humidity = y["humidity"]


    data = [[n,p,k,temperature,humidity,ph,r]]
    ypred = bst.predict(data)
    i=np.argmax(ypred)
    recommendation= crop_categories[i]
    print("Sending result to clint...")
    return recommendation

"""@app.route("/disease")
def upload():

    f= request.args.get('f')
    base= (urllib.parse.unquote(f))
    img=base64.b64decode(base)
    img_file = open('image.jpeg', 'wb')
    img_file.write(img)
    x=image.load_img("image.jpg",target_size=(224,224))
    x=image.img_to_array(x)
    x =x/255
    x=np.expand_dims(x,axis=0)
    p=model.predict(x)
    a=np.argmax(p)
    prediction =disease_categories[a]
    print("Sending result to clint...")
    print(prediction)
    return prediction


def base64_url_decode(f):
        base = f.replace(".", "+" )
        base = f.replace("_", "/" )
        base = f.replace("-", "=" )
        return base64.b64decode(base)
        #(), '._-', '+/='))
"""

if __name__ =='__main__':
	#from waitress import serve
	app.run(host='0.0.0.0', port=8080)