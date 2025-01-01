from django.db import models
from django.utils import timezone
from django.utils.text import slugify

# Create your models here.
class Article(models.Model):
    id = models.IntegerField(primary_key=True)
    title = models.CharField(max_length=100)
    # The slug field is used within the articles url:
    slug = models.SlugField(unique=True, blank=True)
    date = models.DateTimeField(default=timezone.now)
    description = models.TextField(default="Description")
    content = models.TextField()

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.title)
        super(Article, self).save(*args, **kwargs)
    
    def get_absolute_url(self):
        return f"/article/{self.slug}"

    def __repr__(self):
        return f"Article: {self.title}, from date {self.date}"