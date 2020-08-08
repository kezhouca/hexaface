"""hexaface URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.10/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from faceapp import views

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^$',views.index, name='index'),
    url(r'^FaceDetection/',views.FaceDetection, name='FaceDetection'),
    url(r'^engine_1/',views.engine_1, name='engine_1'),
    url(r'^GenderPrediction/',views.GenderPrediction, name='GenderPrediction'),
    url(r'^engine_2/',views.engine_2, name='engine_2'),
    url(r'^AgeEstimation/',views.AgeEstimation, name='AgeEstimation'),
    url(r'^engine_3/',views.engine_3, name='engine_3'),
    url(r'^FacialEmotionRecognition/',views.FacialEmotionRecognition, name='FacialEmotionRecognition'),
    url(r'^engine_4/',views.engine_4, name='engine_4'),
    url(r'^FaceVerification/',views.FaceVerification, name='FaceVerification'),
    url(r'^engine_5/',views.engine_5, name='engine_5'),
    url(r'^FaceGeneration/',views.FaceGeneration, name='FaceGeneration'),
    url(r'^engine_6/',views.engine_6, name='engine_6'),
]

urlpatterns += static( settings.PIC_URL, document_root = settings.PIC_ROOT )
