from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.homepage, name="homepage"),
    path('register/', views.register, name="signup"),
    path('detect/', views.start_detection, name="detect"),
    path('results/', views.results, name="results"),
    path('video/', views.video, name="video"),
    path('click-img/', views.clickimage, name="click-image"),
    path('sample/', views.sample, name="sample"),
]