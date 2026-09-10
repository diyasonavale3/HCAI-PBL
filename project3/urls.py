from django.urls import path
from . import views

app_name = 'project3'

urlpatterns = [
    path('', views.index, name='index'),
    path('task5/', views.task5, name='task5'),
    path('task5/reset/', views.task5_reset, name='task5_reset'),
]
