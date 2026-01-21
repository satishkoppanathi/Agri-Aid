from django.urls import path
from home import views

urlpatterns = [
    path('', views.HomePage, name='home'),
    path('signup',views.SignUp,name='signup'),
    path('signin', views.SignIn, name='signin'),
    path('logout', views.LogOut, name='logout'),
    path('cropRecommendation',views.cropRecommendation, name='cropRecommendation'),
    path('fertilizerPredict',views.fertilizerPredict,name='fertilizerPredict'),
    path('diseaseDetection',views.diseaseDetection , name='diseaseDetection'),
    path('about',views.about,name='about'),
    path('contact',views.contact,name='contact'),
    path('services',views.services,name='services'),
    
]