from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from  . import loadModel
# Create your views here.

@csrf_exempt
def index(request):
    if request.method != 'POST':
        return render(request, 'index.html')
    else:
        #img = request.FILES.get('images')
        img = request.POST['images']
        print(img)
        return JsonResponse({'ans': 1})
        #num = loadModel.hand_writing_reco()

