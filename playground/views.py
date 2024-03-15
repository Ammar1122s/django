from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse

from keras.models import load_model
from keras.models import Sequential, Model
# from keras.layers import Dense, LSTM, GRU, Concatenate, Input
import tensorflow as tf
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, OrderClass
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.stream import TradingStream
import alpaca_trade_api as tradeapi

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# Create your views here.

def say_hello(response):
    # return HttpResponse("Hello World!")
    context = {'a': "Hello world!"}
    return render(response, 'new.html', context=context)


def start(response):
    return HttpResponse("Started!!!!")


def get_data(request):
    
    stockName = request.GET.get('stockName')  
    duration = request.GET.get('duration')
    
    if request.method == 'GET':
        def collect_data(Symbol, start_date, end_date):
            stock_data = yf.download(Symbol, start=start_date, end=end_date)
            return stock_data

        # Test the function
        
        current_date_time = datetime.now()
        previous_days = current_date_time - timedelta(days=int(duration))
        
        df = collect_data(stockName, previous_days, current_date_time)
        
        df_close = df["Close"]
        
        data = df_close.values.tolist()
        print(data)
        return JsonResponse({'data': data})
        
        
    

def predict(request):
    if request.method == 'GET':
        stockName = request.GET.get('stockName')  
        import warnings
        warnings.simplefilter(action="ignore", category=FutureWarning)

        model_path = 'playground/model/st_aapl_model.h5'
        model = load_model(model_path)

        def collect_data(Symbol, start_date, end_date):
            stock_data = yf.download(Symbol, start=start_date, end=end_date)
            return stock_data

        # Test the function
        
        df = collect_data(stockName, '2021-01-01', datetime.now())

        def build_training_dataset(input_ds):
            # Create a new dataframe with only the 'Close column
            input_ds.reset_index()
            data = input_ds.drop(["Adj Close"], axis=1)
            # Convert the dataframe to a numpy array
            dataset = data.values
            # Get the number of rows to train the model on
            training_data_len = len(df)
            return data, dataset, training_data_len

        # Test the function
        # df = df.drop(df.columns[0], axis=1)
        training_data_df, training_dataset_np, training_data_len = build_training_dataset(
            df)
        dataset = training_dataset_np
        data = training_data_df

        def scale_the_data(dataset):
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(dataset)
            closing_price_scaling_params = scaler.data_range_[
                3], scaler.data_min_[3]
            return scaler, scaled_data, closing_price_scaling_params

        # Test the function
        scaler, scaled_data, closing_price_scaling_params = scale_the_data(
            training_dataset_np)
        # scaler_test, scaled_data_test, closing_price_scaling_params_test = scale_the_data(training_dataset_np)

        def inverse_transform_closing_price(scaled_price):
            return scaled_price * closing_price_scaling_params[0] + closing_price_scaling_params[1]

        final = inverse_transform_closing_price(scaled_data[:, 3])

        # Create the scaled training data set

        def split_train_dataset(training_data_len):
            train_data = scaled_data[0:int(training_data_len), :]
            # Split the data into x_train and y_train data sets
            x_train = []
            y_train = []
            for i in range(60, len(train_data)):
                x_train.append(train_data[i-60:i, :])
                y_train.append(train_data[i, 3])

            # Convert the x_train and y_train to numpy arrays
            x_train, y_train = np.array(x_train), np.array(y_train)

            # Reshape the data
            x_train = np.reshape(
                x_train, (x_train.shape[0], x_train.shape[1], 5))

            print(x_train.shape)
            # x_train.shape
            return x_train, y_train

        # Test the function
        x_train, y_train = split_train_dataset(training_data_len)
        x_train = x_train[-1].reshape(1, 60, 5)

        pre = model.predict(x_train)

        pre = inverse_transform_closing_price(pre)

        prediction_list = pre.tolist()

        print(prediction_list)

        context = {'a': "Hello new world!"}

        test = "DOne!"
        # return render(request, 'new.html', context=context)
        return JsonResponse({'prediction': prediction_list})

    return render(request, 'new.html')


def predict_order(request):
    if request.method == 'GET':
        price = int(request.GET.get('price'))  
        amount = float(request.GET.get('amount'))
        
        print("In order")

        BASE_URL = 'https://paper-api.alpaca.markets'
        API_KEY = 'PKLWZEIDYZZOMD1JOA7A'
        SECRET_KEY = 'a9K2vNpiToQXQsNkklhKgHLVfdfTLfKTuRtr0fVg'
        ORDERS_URL = '{}/v2/orders'.format(BASE_URL)
        HEADERS = {'APCA-API-KEY-ID': API_KEY, 'APCA-API-SECRET-KEY': SECRET_KEY}

        def calculate_take_profit_and_stop_loss(price):

            take_profit = price * 1.2  # 20% above the actual price
            stop_loss = price * 0.9  # 10% below the actual price

            base_price = float(price)
            stop_loss = min(stop_loss, base_price - 0.01)
            return take_profit, stop_loss
        
        take_profit, stop_loss = calculate_take_profit_and_stop_loss(price)
        
        print(take_profit, stop_loss)
        
        
        def create_order(pred_price,company,test_loss):
            take_profit_price = 55
            stop_loss_price = 34

            trend = 12


            if trend < 0:
                side = 'sell'
            else:
                side = 'buy'
            if side == 'buy':
                order_details = MarketOrderRequest(
                                    symbol= company,
                                    qty = amount,
                                    side = OrderSide.BUY,
                                    time_in_force = TimeInForce.DAY,
                                    order_class = OrderClass.BRACKET,
                                    take_profit=TakeProfitRequest(limit_price=take_profit),
                                    stop_loss=StopLossRequest(stop_price=stop_loss)
                                    )
            elif side == 'sell':
                order_details = MarketOrderRequest(
                                    symbol= company,
                                    qty = price,
                                    side = OrderSide.SELL,
                                    time_in_force = TimeInForce.DAY,
                                    order_class = OrderClass.BRACKET,
                                    take_profit=TakeProfitRequest(limit_price=take_profit),
                                    stop_loss=StopLossRequest(stop_price=stop_loss)
                                    )
            return order_details


        client = TradingClient(API_KEY, SECRET_KEY, paper=True)
        account = dict(client.get_account())
        order_details = create_order(225.123,'AAPL' , 10.5)
        

        order = client.submit_order(order_data= order_details)

        trades = TradingStream(API_KEY, SECRET_KEY, paper=True)
        
        order_details_dict = {
            "qty": order_details.qty,
            "side": order_details.side,
            'stop_loss':order_details.stop_loss.stop_price,
            "symbol":order_details.symbol,
            "take_profit":order_details.take_profit.limit_price,
            "time_in_force":order_details.time_in_force,
            "order_class":order_details.order_class,
        }
        
        order_dict = {
        'order_id': order.client_order_id,
        }
        
        return JsonResponse({'order': order_details_dict})
        
        
        
    return render(request, 'new.html')
        
        
