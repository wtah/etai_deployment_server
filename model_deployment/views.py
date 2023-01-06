import torch
from PIL import Image
from rest_framework import generics
import torch.nn.functional as F

from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from imantics import Polygons, Mask

from model_deployment.serializers import TextPredictionSerializer
from model_deployment.models import TextPrediction

from etai_deployment_server import settings

from transformers import AutoTokenizer, AutoModelForSequenceClassification, DetrImageProcessor, DetrForObjectDetection
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_text_model():
    if settings.INFERENCE_MODE=='text':
        return AutoTokenizer.from_pretrained("distilbert-base-uncased"), \
               AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
    else:
        return None ,None


### Example for Text based deployment
class TextPredictionListCreate(generics.ListCreateAPIView):
    queryset = TextPrediction.objects.all()
    serializer_class = TextPredictionSerializer
    permission_classes = []
    tokenizer, model = get_text_model()

    ### ENTRYPOINT FOR INFERENCE
    def perform_create(self, serializer):
        # Here you get the text string submitted for inference
        prediction = self.infer(serializer.validated_data['sample'])
        serializer.validated_data['prediction'] = prediction
        if settings.DO_SAVE_PREDICTIONS:
            serializer.save()

    def preprocess(self, text):
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

    def infer(self, text):
        encoded_input = self.tokenizer(self.preprocess(text), return_tensors='pt')
        with torch.no_grad():
            output = self.model(**encoded_input)
        scores = output[0][0].detach().numpy().tolist()
        result = F.softmax(output)
        return result

    # def infer_disc(self, text):
    #     encoded_input = self.tokenizer(self.preprocess(text), return_tensors='pt')
    #     with torch.no_grad():
    #         output = self.model(**encoded_input)
    #     scores = output[0][0].detach().numpy().tolist()
    #     if (scores[0] + scores[1]) > 0.5:
    #         return 'hate'
    #     else:
    #         return 'no hate'    