from django.shortcuts import render
from RoboStockApp.forms import UserForm, UserProfileInfoForm

from django.contrib.auth import authenticate, login, logout
from django.http import HttpResponseRedirect, HttpResponse
from django.urls import reverse
from django.contrib.auth.decorators import login_required

import pandas_datareader as web
import json

import math
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import io

# Create your views here.
def index (request):
    return render(request,'RoboStockApp/index.html')

def marketindexes (request):
    """
    This view returns the major stock market indexes
    """
    #h_var : The title for horizontal axis
    h_var = "Date"
    #v_var : The title for horizontal axis
    v_var = "Points"
    #data : A list of list which will ultimated be used
    # to populate the Google chart.
    #S&P 500
    stock_key = '^GSPC'
    df = web.DataReader(stock_key, data_source = 'yahoo', start = '2012-01-01', end = '2020-12-18')
    #Set the 'Date' column equal to the indes
    df['Date'] = df.index
    #Change the timestamp to type string
    df['Date'] = df['Date'].astype(str)
    #Chose only the 'Date' and 'Close' columns
    df = df[['Date','Close']]
    #Convert the data frame to a Python List data type.
    data = df.values.tolist()
    data.insert(0,['Date','Close'])
    #h_var_JSON : JSON string corresponding to  h_var
    #json.dumps converts Python objects to JSON strings
    h_var_JSON = json.dumps(h_var)
    #v_var_JSON : JSON string corresponding to  v_var
    v_var_JSON = json.dumps(v_var)
    #modified_data : JSON string corresponding to  data
    modified_data = json.dumps(data)

    #h_var : The title for horizontal axis
    h_var1 = "Date"
    #v_var : The title for horizontal axis
    v_var1 = "Points"
    #data : A list of list which will ultimated be used
    # to populate the Google chart.
    #Get the stock quote
    stock_key1 = 'NDAQ'
    df1 = web.DataReader(stock_key1, data_source = 'yahoo', start = '2020-01-01', end = '2020-12-18')
    #Set the 'Date' column equal to the index
    df1['Date'] = df1.index
    #Change the timestamp to type string
    df1['Date'] = df1['Date'].astype(str)
    #Chose only the 'Date' and 'Close' columns
    df1 = df1[['Date','Close']]
    #Convert the data frame to a Python List data type.
    data1 = df1.values.tolist()
    data1.insert(0,['Date','Close'])
    #h_var_JSON : JSON string corresponding to  h_var
    #json.dumps converts Python objects to JSON strings
    h_var_JSON1 = json.dumps(h_var1)
    #v_var_JSON : JSON string corresponding to  v_var
    v_var_JSON1 = json.dumps(v_var1)
    #modified_data : JSON string corresponding to  data
    modified_data1 = json.dumps(data1)

    #Finally all JSON strings are supplied to the charts.html using the
    # dictiory shown below so that they can be displayed on the home screen
    return render(request,'RoboStockApp/marketindexes.html',{'values':modified_data,\
         'h_title':h_var_JSON,'v_title':v_var_JSON,'values1':modified_data1,\
              'h_title1':h_var_JSON1,'v_title1':v_var_JSON1})

def setPlt(stock_quote):

    #Get the stock quote
    stock_key = stock_quote
    df = web.DataReader(stock_key, data_source = 'yahoo', start = '2012-01-01', end = '2020-12-18')
    #Get the number of rows and columns in the data set
    df.shape

    #Create a new dataframe with only the 'Close' column
    data = df.filter(['Close'])
    #Convert the dataframe to a numpy array
    dataset = data.values
    #Get the number of rows to train the LSTM model on. This is taking 80% of the entries. What happens when you take all of it?
    training_data_len = math.ceil(len(dataset)*.8)
    training_data_len

    #Scale the data, Why scale? in practice it is nearly always advantagoues to apply pre processing or scaling of the data before
    #presenting it to the neural network. What happens if you don't scale the data?
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    scaled_data

    #Create the training data set
    #Create the scaled training data set
    train_data = scaled_data[0:training_data_len , :]
    #Split the data into x_train and y_train data sets

    #Independent Training Variable is x_train, this is the historical data.
    x_train = []

    #Dependent variable or Target Variable is y_train, this is the prediction.
    y_train = []

    for i in range(60, len(train_data)):
      x_train.append(train_data[i-60:i, 0])
      y_train.append(train_data[i,0])
      if i <= 60:
        print("x_train:", x_train)
        print("y_train:", y_train)
        print()

    #Convert the x_training and y_training to numpy
    #A python list is a collection of values, it is iterative, it holds different data types, elements can be added, removed, or changed.
    #Operations with python lists are difficult (see example below), it is beneficial to turn these arrays to numpy arrays.
    x_train, y_train = np.array(x_train), np.array(y_train)

    #Reshape the data - the LSTM needs three dimensitional data. x_train and y_train are two dimensional.
    #We need the number of smaples, number of time steps, and the number of features.
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_train.shape

    #This is the model architecture!
    #Build the LSTM model
    model = Sequential()

    #the LSTM has 50 newrons, then a second LSTM layer so True, and the input shape is the # of time steps (60) and # features (1)
    model.add(LSTM(50,return_sequences = True, input_shape = (x_train.shape[1],1)))

    #This is the second layer. It also has 50 newrons, but the return is false (no other/more LSTM layers)
    model.add(LSTM(50, return_sequences = False))

    #This is the desnly connected neural network layer with 2 neurons
    model.add(Dense(25))
    model.add(Dense(1))

    #Compile the model
    #An optimizer is used to improve upon the loss function
    #The loss function quantifies how well it did
    model.compile(optimizer = 'adam', loss='mean_squared_error')

    #Train the model - MACHINE LEARNING!! :-)
    #What is epochs? the number of times a data set is passed forward and backwards through the model
    model.fit(x_train, y_train, batch_size = 1, epochs = 2)

    #Create the testing data set
    #Create a new array containing scaled values from index 1543 to 2003 (why 1543 to 2003? bc that's the end of dataset)
    test_data = scaled_data[training_data_len - 60: , :]
    #Create the data sets x_test and y_test
    x_test = []

    #prediction values are y_test
    y_test = dataset[training_data_len: , :]


    for i in range(60 , len(test_data)):
      x_test.append(test_data[i-60:i,0])

    #Convert the data to a numpy array
    x_test = np.array(x_test)
    x_test.shape

    #Reshape the data to be three dimensional
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    x_test.shape

    #Get the models predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    #Get the root mean squared error (RMSE)
    rsme = np.sqrt( np.mean(predictions - y_test )**2 )
    rsme

    #Plot the data
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions

    #Visualize the data
    plt.figure(figsize = (16,8))
    plt.title('AAPL (Apple Inc.) Stock Price')
    plt.xlabel('Date', fontsize = 18)
    plt.ylabel('Close Price USD ($)' , fontsize = 18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Historical Stock Price Data','Actual Values','SLTM Predictions'], loc = 'lower right')


#This view converts the plot to an SVG
def pltToSvg():
    buf = io.BytesIO()
    plt.savefig(buf, format='svg', bbox_inches='tight')
    s = buf.getvalue()
    buf.close()
    return s


def MLpredictions (request):

    if request.method == 'POST':

        stock_quote = request.POST.get('stock_quote')

        setPlt(stock_quote) # create the plot
        svg = pltToSvg() # convert plot to SVG
        plt.cla() # clean up plt so it can be re-used
        response = HttpResponse(svg, content_type='image/svg+xml')
        return response
    else:
        return render(request, 'RoboStockApp/mlpredictions.html')

# Do not use "def logout()" since logout is an imported module from line 4
@login_required
def user_logout(request):
    # Log the user out
    logout(request)
    # Return to homepage
    return HttpResponseRedirect(reverse('index'))

def register(request):

    registered = False

    if request.method == 'POST':
        user_form = UserForm(data=request.POST)
        profile_form = UserProfileInfoForm(data=request.POST)

        if user_form.is_valid() and profile_form.is_valid():

            user = user_form.save()
            user.set_password(user.password)
            user.save()

            profile = profile_form.save(commit=False)
            profile.user = user

            if 'profile_pic' in request.FILES:
                profile.profile_pic = request.FILES['profile_pic']

            profile.save()

            registered = True

        else:
            print(user_form.errors,profile_form.errors)

    else:
        user_form = UserForm()
        profile_form = UserProfileInfoForm()

    return render(request,'RoboStockApp/registration.html',
                            {
                            'user_form':user_form,
                            'profile_form':profile_form,
                            'registered':registered
                            })

# Do not use "def login()" since logout is an imported module from line 4
def user_login(request):

    if request.method == 'POST':
        #First get the username and password supplied
        username = request.POST.get('username')
        password = request.POST.get('password')

        #Django's built in authentication function:
        user = authenticate(username=username,password=password)

        #If we have a username
        if user:
            #Check that the account is is_active
            if user.is_active:
                #Log the user in
                login(request,user)
                #Send the user back to some page, in this case
                #the home homepage
                return HttpResponseRedirect(reverse('index'))
            else:
                return HttpResponse("Your account is not active.")
        else:
            print("Someone tried to login and failed.")
            print("They used username: {} and password: {}".format(username,passoword))
            return HttpResponse("Invalid login details supplied.")
    else:
        #Nothing has been provided for the username and password
        return render(request, 'RoboStockApp/login.html',{})
