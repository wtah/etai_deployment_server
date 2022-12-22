from rest_framework import serializers
from transformers import AutoTokenizer, AutoModel

from model_deployment.models import TextPrediction,ImagePrediction


class TextPredictionSerializer(serializers.ModelSerializer):

    class Meta:
        model = TextPrediction
        fields = ['sample', 'prediction']



class ImagePredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImagePrediction
        fields = ['sample', 'prediction']
