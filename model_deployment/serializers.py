from rest_framework import serializers
from transformers import AutoTokenizer, AutoModel

from model_deployment.models import TextPrediction


class TextPredictionSerializer(serializers.ModelSerializer):

    class Meta:
        model = TextPrediction
        fields = ['sample', 'prediction']