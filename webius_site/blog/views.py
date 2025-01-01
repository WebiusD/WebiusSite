from django.http import HttpResponse, HttpResponseNotAllowed, JsonResponse, HttpResponseRedirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse
from .models import Article
import json

from .forms import ArticleForm
from .util import get_prev_next_id, get_context, get_prev_next_article_by_slug, convert_markdown
import code
# Create your views here.

def home(request):
    # get the most recent article and render into post template
    latest_article = Article.objects.last()

    prev_id, current_id, next_id = get_prev_next_id(latest_article.id) 
    return render(request, 'blog/post.html', get_context(prev_id, current_id, next_id))

def articles(request):
    # list all articles
    return render(request, 'blog/articles.html', {"articles" : Article.objects.all().order_by('-date')})

def article(request, slug):
    context = get_prev_next_article_by_slug(slug)

    return render(request, 'blog/post.html', context=context)
     #, "next_id": next_id, "next_title": next_title, "prev_id": prev_id, "prev_title": prev_title})

def about(request):
    return render(request, 'blog/about.html')

# write and edit are similar. They only differ in the url their form-action points to:
# create-article for write
# update-article for edit
@login_required
def write(request):
    initial_data = {
        'title': request.POST.get('title', ''),
        'content': request.POST.get('content', '')
    }

    form = ArticleForm(initial=initial_data)

    form_action = reverse('create-article')
    return render(request, 'blog/write.html', {
        "form_action": form_action,
        "form": form
    })
# @login_required
# def write(request):
#     form_action = reverse('create-article')
#     return render(request, 'blog/write.html', {"form_action": form_action, "title": "", "content": ""})

@login_required
def edit_article(request, slug):
    article_to_edit = Article.objects.filter(slug=slug).first()
    form_action = reverse('update-article', args=[slug])

    return render(request, 'blog/write.html', {"form_action": form_action, "title": article_to_edit.title, "content": article_to_edit.content})

@login_required
def create_article(request):
    print("creating article")

    if request.method == 'POST':
        form = ArticleForm(request.POST)

        if form.is_valid():
            article = form.save()
            messages.success(request, 'Article submitted successfully', extra_tags='alert-success')
            return redirect('blog-article', slug=article.slug)  # Redirect to the article detail page
        else:
            messages.error(request, 'Error submitting article', extra_tags='alert-danger')
            return render(request, 'blog/write.html', {'form': form})
    else:
        form = ArticleForm()  # Initialize an empty form for GET requests

    print("Returning to write...")
    return render(request, 'blog/write.html', {'form': form})
    #     title = request.POST.get('title')
    #     content = request.POST.get('content')

    #     converted_content = convert_markdown(content)

    #     article = Article(title=title, content=converted_content)
    #     article.save()

    #      # Display success message:
    #     messages.success(request, 'Article submitted successfully', extra_tags='alert-success')
    #     return redirect('blog-article', slug=article.slug)  # Redirect to the article detail page
    # else:
    #     return HttpResponseNotAllowed(['POST'])

@login_required
def update_article(request, slug):
    # Retrieve the existing article object
    article = get_object_or_404(Article, slug=slug)
    
    if request.method == 'POST':
        form = ArticleForm(request.POST, instance=article)

        if form.is_valid():
            form.save()
            messages.success(request, 'Article updated successfully', extra_tags='alert-success')
            return redirect('blog-article', slug=article.slug)
        else:
            messages.error(request, "Error updating article", extra_tags='alert-danger')
    
    else:
        form = ArticleForm(instance=article)

    return render(request, 'blog/write.html', {"form": form})
    #     title = request.POST.get('title')
    #     content = request.POST.get('content')

    #     converted_content = convert_markdown(content)
        
    #     # Update the article object with new data
    #     article.title = title
    #     article.content = converted_content
    #     article.save()

    #     # Display success message:
    #     messages.success(request, 'Article updated successfully', extra_tags='alert-success')
        
    #     # Redirect to the updated article detail page
    #     return redirect('blog-article', slug=article.slug)
    # else:
    #     return HttpResponseNotAllowed(['POST'])

@login_required
def convert(request):
    print("Got conversion request")
    
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))
            markdown = data.get('markdown')

            print("Original markdown: ", markdown)
            conversion_result = convert_markdown(markdown)

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)

    data = {'converted': conversion_result}
    return JsonResponse(data)

def chat_gpt(request):
    return HttpResponseRedirect('https://chatgpt.com/')