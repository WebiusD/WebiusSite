from django.contrib import admin
from .models import Article, Category

# Register your models here.
# register the Article Model onto the admin site!
admin.site.register(Article)
admin.site.register(Category)