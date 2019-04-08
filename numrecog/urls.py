from django.conf.urls import url
from . import views
#用于djangos
urlpatterns = [
    url(r'^hand$', views.index, name='index'),

]