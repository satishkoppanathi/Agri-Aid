from django.shortcuts import render ,redirect
# for authentication
from django.contrib.auth.models import User 
from django.contrib.auth import authenticate
from django.contrib.auth import logout ,login

from datetime import datetime
from home.models import Contact
from django.contrib import messages

# load models
from joblib import load
cropPredictModel= load('./savedModels/crop_recommendation_model.joblib')
fertilizerPredictModel = load('./savedModels/fertilizerpredict.joblib')
import os
from PIL import Image
import numpy as np
from .forms import PlantUploadForm
from tensorflow import keras
# from django.core.files.storage import FileSystemStorage
from keras.preprocessing import image as keras_image 
from django.conf import settings

plant_disease_model_path = os.path.join(settings.BASE_DIR, 'savedModels', 'plantDiseaseDetect.h5')
plantDiseaseModel = keras.models.load_model(plant_disease_model_path)

# Create your views here.
def HomePage(request):
    if request.user.is_anonymous:
        return redirect("/signup")
    return render(request,'index.html')

def SignUp(request):
    if request.method == 'POST':
        fname = request.POST.get('fname')
        lname = request.POST.get('lname')
        uname = request.POST.get('username')
        email = request.POST.get('email')
        password1 = request.POST.get('password1')
        password2 = request.POST.get('password2')
        user = User.objects.create_user(uname,email,password1)
        user.first_name = fname
        user.last_name = lname
        user.save()
        return redirect('signin')
    return render(request,'login.html')

def SignIn(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        print(username,password)
        # check if user has entered correct credentials
        user = authenticate(username=username, password=password)
        if user is not None:
            login(request,user)
            # A backend authenticated the credentials
            return redirect("/")
        else:
            # No backend authenticated the credentials
            return render(request,'login.html')
    return render(request,'login.html')

def LogOut(request):
    logout(request)
    return redirect("/signin")

def cropRecommendation(request):
    if request.method == 'POST':
        nitrogen = request.POST['nitrogen']
        phosphorous = request.POST['phosphorous']
        potassium = request.POST['potassium']
        temperature = request.POST['temperature'] 
        humidity = request.POST['humidity']
        ph = request.POST['ph']
        rainfall = request.POST['rainfall']
        y_pred = cropPredictModel.predict([[nitrogen,phosphorous,potassium,temperature,humidity,ph,rainfall]])
        crop =" ".join(map(str, y_pred))
        # print(y_pred)
        # print(crops)
        return render(request,'cropRecommendation.html',{'result':crop})
    return render(request,'cropRecommendation.html')


import pandas as pd
def fertilizerPredict(request):
    if request.method == 'POST':
        temperature = request.POST['temperature'] 
        humidity = request.POST['humidity']
        moisture = request.POST['moisture']
        soilType=request.POST['soilType']
        cropType=request.POST['cropType']
        nitrogen = request.POST['nitrogen']
        potassium = request.POST['potassium']
        phosphorous = request.POST['phosphorous']
        print(temperature,humidity,moisture,soilType,cropType,nitrogen,potassium,phosphorous)
        data = pd.DataFrame([[temperature,humidity,moisture,soilType,cropType,nitrogen,potassium,phosphorous]], 
                        columns=['Temparature', 'Humidity', 'Moisture', 'Soil_Type', 'Crop_Type', 'Nitrogen', 'Potassium', 'Phosphorous'])

        predicted_fertilizer = fertilizerPredictModel.predict(data)
        predicted_fertilizer =" ".join(map(str, predicted_fertilizer))
        print(f"Predicted Fertilizer: {predicted_fertilizer[0]}")
        return render(request,'fertilizerPredict.html',{'predicted_fertilizer':predicted_fertilizer})
    return render(request,'fertilizerPredict.html')


   

# labels
CATEGORY_LABELS = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

def preprocess_image(img, target_size):
    """Preprocess the uploaded image to match the model's input requirements."""
    img = img.resize(target_size)  # Resize the image to the target size (e.g., 224x224)
    img_array = keras_image.img_to_array(img)  # Convert image to a NumPy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
    return img_array

def diseaseDetection(request):    
    if request.method == 'POST':
        form = PlantUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                # Get the uploaded image
                img = form.cleaned_data['image']
                # Open the image using PIL
                img = Image.open(img)
                # Preprocess the image to fit the model's input requirements (224x224 is common)
                preprocessed_image = preprocess_image(img, target_size=(224, 224))
                # Make a prediction using the loaded model
                prediction_array = plantDiseaseModel.predict(preprocessed_image)
                # Get the index of the highest probability class
                predicted_class_index = np.argmax(prediction_array, axis=1)[0]
                # Get the predicted class label based on the index
                prediction = CATEGORY_LABELS[predicted_class_index]
                print(prediction)
                return render(request, 'diseaseDetection.html', {'form': form, 'prediction': prediction})
            except Exception as e:
                prediction = f"Error occurred during processing: {str(e)}"
        else:
            prediction = "Invalid form submission. Please upload a valid image."
        return render(request, 'diseaseDetection.html', {'form': form, 'prediction': prediction})
    
    else:
        form = PlantUploadForm()
        # Render the template and pass the form and prediction result to the context
        return render(request, 'diseaseDetection.html')



def about(request):
    return render(request,'about.html')


def contact(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        desc = request.POST.get('desc')
        contact = Contact(name=name,email=email,desc=desc,date=datetime.today())
        contact.save() 
        messages.success(request, "Your message has been sent..")
    return render(request,'contact.html')


def services(request):
    return render(request,"services.html")

