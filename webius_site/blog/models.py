from django.db import models
from django.utils import timezone
from django.utils.text import slugify

class Category(models.Model):
    name = models.CharField(max_length=50, unique=True)

    def __str__(self):
        return self.name
    

# Create your models here.
class Article(models.Model):
    id = models.IntegerField(primary_key=True)
    title = models.CharField(max_length=100)
    # The slug field is used within the articles url:
    slug = models.SlugField(unique=True, blank=True)
    date = models.DateTimeField(default=timezone.now)
    description = models.TextField(default="", blank=True)
    content = models.TextField()
    read_time = models.IntegerField(default=0)
    categories = models.ManyToManyField(Category, blank=True, related_name="articles")
    previous_article = models.ForeignKey(
        'self', on_delete=models.SET_NULL, related_name='previous_articles', null=True, blank=True, default=None
    )

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.title)

        last_article = Article.objects.last()
        if last_article and last_article != self:
            self.previous_article = last_article

        # Calculate read time based on content length
        # average reading speed
        words_per_minute = 200
        word_count = len(self.content.split())
        self.read_time = (word_count // words_per_minute) or 1

        super(Article, self).save(*args, **kwargs)
    
    def get_absolute_url(self):
        return f"/article/{self.slug}"

    def __repr__(self):
        return f"Article: {self.title}, from date {self.date}"