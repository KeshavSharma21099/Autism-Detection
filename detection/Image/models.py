from django.db import models
from django.contrib.auth.models import User


class Image(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    frame_image = models.ImageField(upload_to='images/')
    verdict = models.IntegerField(default=0)
    name = models.CharField(max_length=100, null=True)
