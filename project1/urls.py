from django.urls import path
from . import views

app_name = 'project1'

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload, name='upload'),
    path('preview/', views.preview, name='preview'),
    path('visualize/', views.visualize, name='visualize'),
]
