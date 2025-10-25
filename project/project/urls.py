"""
URL configuration for project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from myapp import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.RegisterPage, name='register'),
    path('login/',views.LoginPage, name='login'),
    path('logout/',views.LogoutPage, name='logout'),
    path('index/',views.IndexPage, name='index'),
    path('encrypt/',views.EncryptionPage, name='encrypt'),
    path('decryption/',views.DecryptionPage, name='decryption'),
    path('decrypt/', views.DecryptionView, name='decrypt'),
    path('verify_otp/', views.VerifyOTP, name='verify_otp'),
    path('send_otp/', views.SendOTPForImage, name='send_otp'),
    path('show_histogram/', views.HistogramPage, name='show_histogram'),
    path('differentialAttacks/<int:image_id>/', views.DifferentialAttacksPage, name='differentialAttacks'),
    path('bifurcation/',views.BifurcationPage, name='bifurcation'),
    path('lyapunov/', views.LyapunovGraphPage, name='lyapunov'),
    path('generate_key/', views.generate_key, name='generate_key'),
    path('save_key/', views.SaveKey, name='save_key'),
    path('encryptedImages/', views.EncryptedImages, name='encryptedImages'),
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)