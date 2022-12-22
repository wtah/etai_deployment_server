import torch
from PIL import Image
from rest_framework import generics

from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from imantics import Polygons, Mask

from model_deployment.serializers import TextPredictionSerializer, ImagePredictionSerializer
from model_deployment.models import TextPrediction, ImagePrediction

from etai_deployment_server import settings

from transformers import AutoTokenizer, AutoModelForSequenceClassification, DetrImageProcessor, DetrForObjectDetection
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_text_model():
    if settings.INFERENCE_MODE=='text':
        return AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment"), \
               AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    else:
        return None ,None

def get_image_model():
    if settings.INFERENCE_MODE=='image':
        return DetrImageProcessor.from_pretrained("facebook/detr-resnet-50"), \
            DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

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
        return {'labels':scores}





### Example for Image based deployment
class ImagePredictionListCreate(generics.ListCreateAPIView):
    queryset = ImagePrediction.objects.all()
    serializer_class = ImagePredictionSerializer
    permission_classes = []

    extractor, model = get_image_model()

    ### ENTRYPOINT FOR INFERENCE
    def perform_create(self, serializer):
        prediction = self.infer(serializer.validated_data['sample'])
        serializer.validated_data['prediction'] = prediction
        if settings.DO_SAVE_PREDICTIONS:
            serializer.save()

    def preprocess(self, image):
        # Here you load the submitted image
        img = Image.open(self.request.FILES['sample'])
        # Resize image to know dimensions
        img.thumbnail((400,400))
        return img

    def process_logits(self, outputs, image):
        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.extractor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        predictions=[]
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            predictions.append({'label': self.model.config.id2label[label.item()], 'score':round(score.item(), 3), 'box': box })
            print(
                f"Detected {self.model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
            )
        return predictions

    def infer(self, image):
        img = self.preprocess(image)
        inputs = self.extractor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        preds = self.process_logits(outputs, img)
        return {'labels': preds}
