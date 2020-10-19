from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def delete_nan_from_data(data):
    new_data = data.dropna(axis=0, how='any')
    return new_data

def print_data_info(data):
    print(data.describe())
    print(data.info())

def logistic_regression(data):
    X, y= data[['Date', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
                'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm',
                'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
                'Temp3pm', 'RainToday', 'RISK_MM']], data['RainTomorrow']
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf = LogisticRegression(solver='liblinear')
    model = clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print("Classification Report: ", "\n",
          classification_report(y_test, predictions), "\n")