from django.contrib import admin
from .models import Article

# Register your models here.
# register the Article Model onto the admin site!
admin.site.register(Article)