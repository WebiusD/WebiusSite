from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name="blog-home"),
    path('articles', views.articles, name="blog-articles"),
    path('article/<slug:slug>', views.article, name="blog-article"),
    path('about', views.about, name="blog-about"),
    path('write', views.write, name="blog-write"),
    path('edit/<slug:slug>', views.edit_article, name='edit-article'),
    path('create', views.create_article, name="create-article"),
    path('update/<slug:slug>', views.update_article, name="update-article"),
    path('convert', views.convert, name="convert"),
]