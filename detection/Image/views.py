from django.http import HttpResponse
from django.shortcuts import render, redirect
from .forms import *


# Create your views here.
def hotel_image_view(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)

        if form.is_valid():
            form.save()
            return redirect('success')
    else:
        form = ImageForm()
    return render(request, 'time1.html', {'form': form})


def success(request):
    return HttpResponse('successfully uploaded')