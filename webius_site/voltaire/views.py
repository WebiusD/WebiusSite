from django.shortcuts import render
# from django.http import HttpResponse

# Create your views here.
def index(request):
    # no context:
    return render(request, 'voltaire/index.html')