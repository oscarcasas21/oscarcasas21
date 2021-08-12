from django import forms
from django.core.exceptions import ValidationError
from recognizer.models import Person


class RecognizerForm(forms.Form):
    # def validate_name(name):
    #     if Person.objects.filter(name=name).exists():
    #         raise ValidationError('Person with same name already exists')
    #     return name

    # name = forms.CharField(max_length=100, validators=[validate_name])
    name = forms.CharField(max_length=100)
    image = forms.ImageField()
