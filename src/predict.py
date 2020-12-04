from joblib import load
import pandas as pd

def predict():
    chest = input('Body Chest Measurement: ')
    shoulder = input('Shoulder Measurement: ')
    wrist = input('Wrist Measurement: ')
    query = [[chest, shoulder, wrist]]
    from_3 = load('from_3.pkl')
    to_4 = load('to_4.pkl')

    columns = pd.DataFrame(index=list(load('column_names.pkl')))
    columns['predicton'] = pd.DataFrame.fillna

    predictions = to_4.predict(from_3.predict(query))
    for i in range(len(columns)):
        columns.iloc[i] = predictions[0,i]

    return columns

if __name__ == '__main__':
    print(predict())