from django.db import models

# Create your models here.
class Prediction(models.Model):
    created = models.DateTimeField(auto_now_add=True)
    prediction = models.JSONField(blank=True)

class TextPrediction(Prediction):
    sample = models.TextField()

class ImagePrediction(Prediction):
    sample = models.ImageField(upload_to='upload')