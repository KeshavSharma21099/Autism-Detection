from django.shortcuts import render, redirect, HttpResponse
from .forms import ProfileForm, UpdateProfileForm, UserCreateForm
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import UserProfile
from django.contrib.auth.decorators import login_required
from Image.models import Image
import json
from django.http import JsonResponse
import base64
import uuid
from django.core.files.base import ContentFile
from .driver import *
from pathlib import Path

gaze = GazeTracking()


def homepage(request):
    # return HttpResponse('<h1>Home Page</h1>')
    return render(request, "base.html", {'user': request.user})


def results(request):
    global gaze

    print("INSIDE IF CONDITION")
    for i in Image.objects.all():
        id1 = i.id
        if i.verdict == 0:
            path = "media/images/" + str(i.name)
            ans = pic(path, gaze)
            print("ANS WAS " + str(ans))
            Image.objects.filter(id=id1).update(verdict=ans)

    # ret = 0
    right = find_right()
    if right >= 0.4747:
        # Negative
        print("Result = 2")
        return render(request, 'result.html', {'result': 2})
    elif right < 0.2978:
        # Positive
        print("Result = 1")
        return render(request, 'result.html', {'result': 1})
    else:
        # Cannot Determine
        print("Result = 0")
        return render(request, 'result.html', {'result': 0})


def register(request):
    if request.method == 'POST':
        user_form = UserCreateForm(request.POST)
        profile_form = ProfileForm(request.POST)
        if user_form.is_valid() and profile_form.is_valid():
            user_form.save()
            name = profile_form.cleaned_data["name"]
            mail = profile_form.cleaned_data["mail"]
            age = profile_form.cleaned_data["age"]
            gender = profile_form.cleaned_data["gender"]
            user = User.objects.get(username=user_form.cleaned_data["username"])
            p = UserProfile(user=user, name=name, mail=mail, age=age, gender=gender)
            p.save()
            return redirect("/login")
        else:
            return HttpResponse('<h1>Invalid Form</h1>')
    else:
        user_form = UserCreateForm()
        profile_form = ProfileForm()

    # return HttpResponse('<h1>Sign Up Form</h1>')
    return render(request, "register.html", {'user_form': user_form, 'profile_form': profile_form})


def start_detection(request):
    return render(request, "time_1.html")

@login_required
def afterlogin(request):
    return render(request, "after_login.html")


def video(request):
    return render(request, "video_1.html")


def sample(request):
    if request.method == 'POST':
        mydict = dict(request.POST)
        print(mydict['txt'][0])
    return HttpResponse('<h1>Successful</h1>')

@login_required
def clickimage(request):
    if request.method == 'POST':
        # if request.user:
        #     data = dict(request.POST)
        #     img = data['img'][0]
        #     if img is None:
        #         print("It is None")
        #     else:
        #         print("Successful request")
        #     print(data)
        # print(request.POST)
        data = dict(request.POST)['img'][0]
        # print(img_base64)
        # img = base64.b64decode(img_base64)
        # print(img)


        # Check if the base64 string is in the "data:" format
        if 'data:' in data and ';base64,' in data:
            # Break out the header from the base64 content
            header, data = data.split(';base64,')

        # Try to decode the file. Return validation error if it fails.
        try:
            decoded_file = base64.b64decode(data)
        except TypeError:
            TypeError('invalid_image')

        # Generate file name:
        # n = 0
        # if Image.objects.last():
        #     n = Image.objects.last().id
        # 12 characters are more than enough.
        # Get the file name extension:
        name = str(request.user.username) + str(uuid.uuid4())[:6] + ".jpeg"

        # complete_file_name = "%s.%s" % (file_name, file_extension,)

        file = ContentFile(decoded_file, name=name)

        img = Image(user=request.user, frame_image=file, name=name)
        img.save()

        path = '../media/images/' + name
        while Path(path) is None:
            i = i+1
            if i == 100:
                return JsonResponse({'text': "Image was not saved"})

        # ans = pic(path)
        # img.verdict = ans

        if len(Image.objects.all()) % 10 == 0:
            return results(request)
    #
    return JsonResponse({'text': "Got the image"})


def find_right():
    left = 0
    right = 0
    # curr = 0
    linc = 1
    rinc = 1
    for i in Image.objects.all():
        if i.verdict == 1:
            left = left + linc
            linc = linc + 2
            if rinc > 1:
                rinc = rinc - 1
        elif i.verdict == 2:
            right = right + rinc
            rinc = rinc + 2
            if linc > 1:
                linc = linc - 1
        else:
            if linc > 1:
                linc = linc - 1
            if rinc > 1:
                rinc = rinc - 1
        # curr = i.verdict
    if left+right != 0:
        return right/(left+right)
    else:
        return 0
