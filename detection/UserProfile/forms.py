from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from .models import UserProfile


class ProfileForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ('name', 'mail', 'age', 'gender')


class UserCreateForm(UserCreationForm):
    # email = forms.EmailField(required=True)

    def __init__(self, *args, **kwargs):
        super(UserCreateForm, self).__init__(*args, **kwargs)

        for fieldname in ['username', 'password1', 'password2']:
            self.fields[fieldname].help_text = None


class UpdateProfileForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ('name', 'mail', 'age', 'gender')
        widgets = {
            'age': forms.IntegerField()
        }