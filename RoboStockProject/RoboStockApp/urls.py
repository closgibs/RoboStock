from django.urls import path
from . import views

# SET THE NAMMESPACE FOR TEMPLATE TAGGING!!!
app_name = 'RoboStockApp'

urlpatterns = [
#    path('',views.index,name = 'index'),
    path('home/',views.home,name='home'),
    path('marketindexes/',views.marketindexes,name='marketindexes'),
    path('watchlists/',views.watchlists,name='watchlists'),
    path('mlpredictions/',views.MLpredictions,name='MLpredictions'),
    path('machinelearningalgorithm',views.machinelearningalgorithm,name='machinelearningalgorithm'),
    path('user_login/',views.user_login,name='user_login'),
    path('register/',views.register,name='register'),

    path('svg/',views.get_svg,name='AppleStockMLsvg')
    ]
