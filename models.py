from django.db import models
from django.utils import tree


def upload(instance, filename):
    return f'{instance.person.name}/{filename}' if instance.person else f'Unknown/{filename}'


class Person(models.Model):
    name = models.CharField(max_length=100, unique=True)
    count = models.PositiveIntegerField(default=1)

    def __str__(self):
        return self.name


class PersonImage(models.Model):
    image = models.ImageField(upload_to=upload)
    person = models.ForeignKey(Person, on_delete=models.CASCADE, null=True, blank=True)

    def __str__(self):
        return f"{self.person.name} ({self.image.name.split('/')[-1]})" if self.person else self.image.name.split('/')[-1]
