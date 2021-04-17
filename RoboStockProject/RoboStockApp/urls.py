from django.urls import path
from . import views

# SET THE NAMMESPACE FOR TEMPLATE TAGGING!!!
app_name = 'RoboStockApp'

urlpatterns = [
#    path('',views.index,name = 'index'),
    path('marketindexes/',views.marketindexes,name='marketindexes'),
    path('mlpredictions/',views.MLpredictions,name='MLpredictions'),
    path('user_login/',views.user_login,name='user_login'),
    path('register/',views.register,name='register'),
    ]
