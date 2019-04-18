from flask import Flask, render_template,request,flash, url_for, request, session, redirect
import requests
import json
import calendar
import datetime
import pandas as pd #for data manipulation
import numpy as np #for converting data into array
from sklearn.model_selection import train_test_split #for training model & sepration of data & training set
from sklearn.ensemble import RandomForestRegressor #importing the model we are using
import random

app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def index():
    api_key = 'bfb6a971627fec5e4c8c677acf90e5e0'
    api_call = 'https://api.openweathermap.org/data/2.5/forecast?appid=' + api_key

    if request.method == 'POST':
        if request.form['city'] != 'NULL':
            search = 0
            city = request.form['city']
            if city.lower() == 'sf':
                city = 'San Francisco, US'
            api_call += '&q=' + city
            json_data = requests.get(api_call).json()
            location_data = {
                'city': json_data['city']['name'],
                'country': json_data['city']['country']
            }
            print('\n{city}, {country}'.format(**location_data))
            current_date = ''
            for item in json_data['list']:
                time = item['dt_txt']
                next_date, hour = time.split(' ')
                if current_date != next_date:
                    current_date = next_date
                    year, month, day = current_date.split('-')
                    date = {'y': year, 'm': month, 'd': day}
                print('\n{m}/{d}/{y}'.format(**date))
                hour = int(hour[:2])
                if hour < 12:
                    if hour == 0:
                        hour = 12
                    meridiem = 'AM'
                else:
                    if hour > 12:
                        hour -= 12
                    meridiem = 'PM'
                print('\n%i:00 %s' % (hour, meridiem))
                temperature = item['main']['temp']
                temperature = temperature - 273.15
                description = item['weather'][0]['description'],
                # print('Weather condition: %s' % description)
                # print('Celcius: {:.2f}'.format(temperature - 273.15))
                # print('Farenheit: %.2f' % (temperature * 9/5 - 459.67))
            # mydict = {k: unicode(v).encode('utf-8') for k,v in json_data.items()}
            return render_template('index.html', city_data = json_data)
    else:
        return render_template('index.html')

@app.route('/model', methods=['GET','POST'])
def model():
    if request.method == 'POST':
        date = request.form['date']
        year = date.split('-')[0]
        features = pd.read_csv('/Users/abcd/Desktop/Weather-prediction/temps.csv')
        features.head(5)
        features.describe()
        features = pd.get_dummies(features)
        labels = np.array(features['actual'])
        features= features.drop('actual', axis = 1)
        feature_list = list(features.columns)
        features = np.array(features)
        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
        baseline_preds = test_features[:, feature_list.index('average')]
        baseline_errors = abs(baseline_preds - test_labels)
        rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
        rf.fit(train_features, train_labels)
        temp1 = random.randint(1,60)
        temp2 = random.randint(1,70)
        actual = random.randint(1,60)
        test = [year,10.0,28.0,temp1,temp2,actual,65.0,74.0,53.0,45.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0]
        predictions = rf.predict([test])
        print('Predicted Temperature:',predictions[0])
        errors = abs(predictions - test_labels)
        mape = 100 * (errors / test_labels)
        accuracy = 100 - np.mean(mape)
        print('Accuracy:', round(accuracy, 2), '%.')
        return render_template('result.html', pre = predictions[0], acc=round(accuracy,2),date=date)
    else:
        return render_template('index.html')
@app.route('/contact', methods=['GET','POST'])
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)