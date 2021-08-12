from django.urls import path
from recognizer.views import AboutView, ContactUsView, HomeView, ImageRecognizerView


urlpatterns = [
    path('', HomeView.as_view(), name='home'),
    path('about/', AboutView.as_view(), name='about'),
    path('find_loved_ones/', ImageRecognizerView.as_view(), name='find_loved_ones'),
    path('contact_us/', ContactUsView.as_view(), name='contact_us'),
]
