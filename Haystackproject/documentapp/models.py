from django.db import models
from django import forms
# Create your models here.


class FileUploadForm(forms.Form):
    file = forms.FileField()