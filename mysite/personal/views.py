from django.shortcuts import render

def index(request):
    return render(request, 'personal/home.html')
def sentiment(request):
    return render(request, 'personal/sentiment.html')
def electoral(request):
    return render(request, 'personal/electoral.html')