# Generated by Django 5.0.4 on 2025-01-02 08:49

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('blog', '0008_article_next_article_article_previous_article'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='article',
            name='next_article',
        ),
        migrations.AlterField(
            model_name='article',
            name='previous_article',
            field=models.ForeignKey(blank=True, default=None, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='previous_articles', to='blog.article'),
        ),
    ]