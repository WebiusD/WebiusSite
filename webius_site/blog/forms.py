from django import forms
from .models import Article

from .util import convert_markdown


class ArticleForm(forms.ModelForm):
    class Meta:
        model = Article
        fields = ['title', 'content']

    def clean_content(self):
        content = self.cleaned_data['content']
        converted_content = convert_markdown(content)

        if not converted_content.strip():
            raise forms.ValidationError("Content cannot be empty.")

        return converted_content

    # def clean_title(self):
    #     title = self.cleaned_data['title']
        
    #     # Check if title is empty
    #     if not title.strip():
    #         raise forms.ValidationError("Title cannot be empty.")
        
    #     return title