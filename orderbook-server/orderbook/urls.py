from django.urls import path
from orderbook import views


urlpatterns = [
    path('', views.index, name='index'),
]
