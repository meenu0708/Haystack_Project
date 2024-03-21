

from django.urls import path,include

from .import views

urlpatterns = [

    path('',views.index,name='index'),
    path('upload/', views.upload_file, name='upload_file'),
    path('upload/query', views.query, name='query'),

]
