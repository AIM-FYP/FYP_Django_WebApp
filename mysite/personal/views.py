from django.shortcuts import render
import json
from personal.classification_model import *


def index(request):
    return render(request, 'personal/home.html')
def sentiment(request):
    my_dict =[{
                'one':[["bad"],["neg"]],
                'two':[["good"],["pos"]]
             },
             {
                'one':[["bad"],["neg"]],
                'two':[["good"],["pos"]]
             }]
    js_data = json.dumps(my_dict)
    return render(request, 'personal/sentiment.html',{"content":js_data})
def electoral(request):
    return render(request, 'personal/electoral.html')