from django.urls import path
from . import views

app_name = 'project4'

urlpatterns = [
    path('', views.index, name='index'),
    path('start/', views.start_study, name='start_study'),
    path('pairwise/', views.pairwise, name='pairwise'),
    path('ranking/', views.ranking, name='ranking'),
    path('complete/', views.complete, name='complete'),
]
