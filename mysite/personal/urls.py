from django.conf.urls import url, include
from . import views
from django.urls import path



urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^(\d+)/$', views.index, name='index'),
    url(r'^(\d+)$', views.index, name='index'),
    url(r'^home$', views.index, name='index'),
    url(r'^home/(\d+)/$', views.index, name='index'),
    url(r'^home/(\d+)$', views.index, name='index'),
    url(r'^index$', views.index, name='index'),
    url(r'^index/(\d+)$', views.index, name='index'),
    url(r'^index/(\d+)/$', views.index, name='index'),
    url(r'^electoral$', views.electoral, name='electoral'),
]
