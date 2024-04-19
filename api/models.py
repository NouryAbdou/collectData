from django.db import models

class TrainingData(models.Model):
    x = models.FloatField()
    y = models.FloatField()