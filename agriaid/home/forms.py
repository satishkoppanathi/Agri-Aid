from django import forms
from .models import Image

class PlantUploadForm(forms.Form):
    image = forms.ImageField()