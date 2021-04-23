from django.test import TestCase
from RoboStockApp.views import index

# Create your tests here.

#https://test-driven-django-development.readthedocs.io/en/latest/03-views.html
class ProjectTests(TestCase):

    def test_main_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'RoboStockApp/index.html')
        self.assertContains(response, 'Welcome To RoboStock!')

    def test_home_page(self):
        response = self.client.get('/RoboStockApp/home/')
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'RoboStockApp/home.html')
        self.assertContains(response, "Web application's Home Page and index")

    def test_marketindexes_page(self):
        response = self.client.get('/RoboStockApp/marketindexes/')
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'RoboStockApp/marketindexes.html')
        self.assertContains(response, 'Major Stock Exchange Indexes')

    def test_mlpredictions_page(self):
        response = self.client.get('/RoboStockApp/mlpredictions/')
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'RoboStockApp/mlpredictions.html')
        self.assertContains(response, 'Machine Learning Stock Price Predictions')

    def test_userlogin_page(self):
        response = self.client.get('/RoboStockApp/user_login/')
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'RoboStockApp/login.html')
        self.assertContains(response, 'Please Login')

    def test_register_page(self):
        response = self.client.get('/RoboStockApp/register/')
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'RoboStockApp/registration.html')
