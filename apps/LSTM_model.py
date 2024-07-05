from flask import *
from flask_cors import cross_origin
import io
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math 
from tensorflow.keras.models import load_model

LSTM_module = Blueprint('LSTM_model',__name__)

@LSTM_module.route('/Generate-actual-predicted-price-lstm',methods=['GET'])
@cross_origin(supports_credentials=True)
def Generate_ap_price():
    LSTM_model =  load_model('LSTM_model.h5')
    scaler_X = pickle.load(open('scaler_X.pkl', 'rb'))
    scaler_Y = pickle.load(open('scaler_Y.pkl', 'rb'))

    csv_data = session.get('csv_data')
    current_app.logger.debug(f'CSV data from session: {csv_data[:100]}')
    if not csv_data:
        return jsonify({'message': 'No CSV data found in session'}), 400
    
    df = pd.read_csv(io.StringIO(csv_data), encoding='unicode_escape')

    #preprocessing the data before fitting into the model
    X = df[['RSI', "SMA", "EMA"]]
    y = df['Close']

    X_test = np.array(X)
    X_test = scaler_X.transform(X_test)

    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    y_test = np.array(y).reshape(-1, 1)


    predictions_scaled = LSTM_model.predict(X_test).reshape(-1, 1)
    predictions = scaler_Y.inverse_transform(predictions_scaled)

    date = np.array(df['Date'])
    
    plt.figure(facecolor='lightblue')
    plt.plot( y_test, label='Actual', color='blue')
    plt.plot( predictions, label='Predicted', color='green')
    plt.title("Actual vs Predicted Price")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual", "Predicted"])

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # Send the image as a response
    return send_file(img, mimetype='image/png')


@LSTM_module.route('/Generate-feature-importance-lstm', methods = ['GET'])
def feature_importance():
    LSTM_model =  load_model('LSTM_model.h5')
    scaler_X = pickle.load(open('scaler_X.pkl', 'rb'))
    scaler_Y = pickle.load(open('scaler_Y.pkl', 'rb'))

    csv_data = session.get('csv_data')
    current_app.logger.debug(f'CSV data from session: {csv_data[:100]}')
    if not csv_data:
        return jsonify({'message': 'No CSV data found in session'}), 400
    
    df = pd.read_csv(io.StringIO(csv_data), encoding='unicode_escape')

    #preprocessing the data before fitting into the model
    X = df[['RSI', "SMA", "EMA"]]
    y = df['Close']

    X_test = np.array(X)
    X_test = scaler_X.transform(X_test)

    y_test = np.array(y)
    X_test = X_test.reshape(-1,3)

    result = permutation_importance(LSTM_model, X_test, y_test, n_repeats=10, random_state=42, scoring = "r2")

    # Get the importance scores
    importance_scores = result.importances_mean
    importance_std = result.importances_std

    # Create a dataframe to display the feature importances
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importance_scores,
        'Std Dev': importance_std
    })

    plt.figure(facecolor='lightblue')
    bars = plt.barh(feature_importance_df['Feature'], 
             feature_importance_df['Importance'], 
             label = "Feature Importance")
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.title('Feature Importance based on Permutation Importance')
    plt.legend()
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, f'{width:.2f}', ha='left', va='center')
    plt.gca().invert_yaxis()


    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # Send the image as a response
    return send_file(img, mimetype='image/png')


@LSTM_module.route('/Generate-correlation-lstm')
def correlation():
    LSTM_model =  load_model('LSTM_model.h5')
    scaler_X = pickle.load(open('scaler_X.pkl', 'rb'))
    scaler_Y = pickle.load(open('scaler_Y.pkl', 'rb'))

    csv_data = session.get('csv_data')
    current_app.logger.debug(f'CSV data from session: {csv_data[:100]}')
    if not csv_data:
        return jsonify({'message': 'No CSV data found in session'}), 400
    
    df = pd.read_csv(io.StringIO(csv_data), encoding='unicode_escape')

    #preprocessing the data before fitting into the model
    X = df[['RSI', "SMA", "EMA"]]
    y = df['Close']

    X_test = np.array(X)
    X_test = scaler_X.transform(X_test)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    y_test = np.array(y).reshape(-1, 1)

    predictions = LSTM_model.predict(X_test).reshape(-1, 1)
    
    X_test = X_test.reshape(-1,3)
    # Create a DataFrame from X_test for correlation calculation
    X_test_df = pd.DataFrame(X_test, columns=['RSI', 'SMA', 'EMA'])

    # Add predictions as a new column
    X_test_df['predictions'] = predictions

    # Calculate Pearson correlation coefficients
    correlation_matrix = X_test_df.corr()

    # Display the correlation of each feature with the predictions
    correlation_with_predictions = correlation_matrix['predictions'].drop('predictions')
    print(correlation_with_predictions)

    plt.figure(facecolor='lightblue')
    bars = plt.barh(correlation_with_predictions.index, correlation_with_predictions.values , label="Correlation")
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Feature')
    plt.title('Feature Correlation with Predictions')
    plt.legend()
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, f'{width:.2f}', ha='left', va='center')
    plt.gca().invert_yaxis()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # Send the image as a response
    return send_file(img, mimetype='image/png')

@LSTM_module.route('/Generate-evaluation-metrics-lstm')
def evaluation():
    LSTM_model =  load_model('LSTM_model.h5')
    scaler_X = pickle.load(open('scaler_X.pkl', 'rb'))
    scaler_Y = pickle.load(open('scaler_Y.pkl', 'rb'))

    csv_data = session.get('csv_data')
    current_app.logger.debug(f'CSV data from session: {csv_data[:100]}')
    if not csv_data:
        return jsonify({'message': 'No CSV data found in session'}), 400
    
    df = pd.read_csv(io.StringIO(csv_data), encoding='unicode_escape')

    #preprocessing the data before fitting into the model
    X = df[['RSI', "SMA", "EMA"]]
    y = df['Close']

    X_test = np.array(X).reshape(-1,3)
    
    X_test = scaler_X.transform(X_test)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    y_test = np.array(y).reshape(-1, 1)
    y_test = scaler_Y.transform(y_test)


    predictions = LSTM_model.predict(X_test)
    predictions = np.array(predictions).reshape(-1, 1)
    y_test = y_test.reshape(-1,1)

    mse = mean_squared_error(y_test,predictions)

    rmse = math.sqrt(mse)

    mae = mean_absolute_error(y_test, predictions)

    r2 = r2_score(y_test, predictions)

    evaluation_metrics = {
        'Metric': ['MSE', 'RMSE', 'MAE', 'R2'],
        'Value': [mse, rmse, mae, r2]
    }
    evaluation_df = pd.DataFrame(evaluation_metrics)

    plt.figure(facecolor='lightblue')
    bars=  plt.barh(evaluation_df['Metric'], evaluation_df['Value'] , label="Evaluation metrics")
    plt.xlabel('Evaluation values')
    plt.ylabel('metrics')
    plt.title('Evaluation results of LSTM Model')
    plt.legend()

    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, f'{width:.2f}', ha='left', va='center')
    plt.gca().invert_yaxis()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # Send the image as a response
    return send_file(img, mimetype='image/png')