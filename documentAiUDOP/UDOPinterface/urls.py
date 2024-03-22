from django.urls import path
from . import views

urlpatterns = [
    path('inference/', views.inference_view, name='inference')
]