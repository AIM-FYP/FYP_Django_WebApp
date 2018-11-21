from django.conf.urls import url, include
from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^sentiment', views.sentiment, name='sentiment'),
    url(r'^electoral', views.electoral, name='electoral'),
]
