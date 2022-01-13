from django.views.generic.base import TemplateView
from recognizer.models import Person, PersonImage
from django.shortcuts import redirect, render
from django.views.generic import View
from recognizer.facenet import HDF5Store
from recognizer.forms import RecognizerForm
from django.core.files.base import ContentFile
from django.contrib import messages
from django.template.loader import render_to_string
from sendgrid.helpers.mail import Mail
from sendgrid import SendGridAPIClient
from face_recognition.settings import SENDGRID_API_KEY, SENDER_EMAIL, RECEIVER_EMAIL


vech5 = HDF5Store('embeddingVec.h5', 'vecs',)


class HomeView(TemplateView):
    template_name = 'recognizer/home.html'


class AboutView(TemplateView):
    template_name = 'recognizer/about.html'


class ContactUsView(View):
    def get(self, request):
        return render(request, 'recognizer/contact_us.html')

    def post(self, request):
        message = render_to_string('email/contact_email.html', {
            'first_name': request.POST.get('first_name', ''),
            'last_name': request.POST.get('last_name', ''),
            'address': request.POST.get('address', ''),
            'email': request.POST.get('email', ''),
            'phone_number': request.POST.get('phone_number', ''),
            'message': request.POST.get('message', ''),
        })
        mail = Mail(from_email=SENDER_EMAIL, to_emails=RECEIVER_EMAIL, html_content=message,
                    subject=f"Contact ({request.POST.get('first_name', '')} {request.POST.get('last_name', '')})")
        try:
            send_grid = SendGridAPIClient(SENDGRID_API_KEY)
            send_grid.send(mail)
            print(send_grid)
        except Exception as e:
            print(e.body)
            pass
        return redirect('home')


class ImageRecognizerView(View):
    def get(self, request):
        return render(request, 'recognizer/find_loved_ones.html', context={'form': RecognizerForm()})

    def post(self, request):
        form = RecognizerForm(request.POST, request.FILES)
        if form.is_valid():
            person_image = PersonImage.objects.create(image=request.FILES['image'])
            name = vech5.getname(person_image.image.path)
            if name is None:
                person, _ = Person.objects.get_or_create(name=form.cleaned_data['name'])
                messages.success(request, 'Person not identified')
            else:
                person, _ = Person.objects.get_or_create(name=name)
                messages.success(request, f'Person identified as {name}')
                person.count += 1
                person.save()
            vech5.addtodb(name=f'{person.id}-{person.name}-{person.count}',
                            path=person_image.image.path)
            person_image.person = person
            person_image.image = ContentFile(person_image.image.read(), name=person_image.image.name.split('/')[-1])
            person_image.save()
            return redirect('find_loved_ones')
        return render(request, 'recognizer/find_loved_ones.html', context={'form': form})
