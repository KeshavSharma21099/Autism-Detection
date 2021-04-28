from django import forms
from .models import *


class ImageForm(forms.ModelForm):
    class Meta:
        model = Image
        fields = ['frame_img']

        widgets = {
            'frame_img': forms.ImageField(attrs={
                'id': 'post-img'
            })
        }
