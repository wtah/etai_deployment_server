"""etai_deployment_server URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from etai_deployment_server import settings
from model_deployment import views

if settings.INFERENCE_MODE == 'text':
    inference_path = path('', views.TextPredictionListCreate.as_view())
elif settings.INFERENCE_MODE == 'image':
    inference_path = path('', views.ImagePredictionListCreate.as_view())


urlpatterns = [
    path('admin/', admin.site.urls),
    inference_path
]
