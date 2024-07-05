from flask import *
from distutils.log import debug
from fileinput import filename
import pandas as pd
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS ,cross_origin
from flask_session import Session

from apps.setup import create_app
app= create_app()
CORS(app,supports_credentials=True)

#Register the model file
from apps.RF_model import RF_module
app.register_blueprint(RF_module)

from apps.GB_model import GB_module
app.register_blueprint(GB_module)

from apps.LSTM_model import LSTM_module
app.register_blueprint(LSTM_module)

from apps.GRU_model import GRU_module
app.register_blueprint(GRU_module)

app.secret_key = 'This is your secret key to utilize session in Flask'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['UPLOAD_FOLDER'] = os.path.join('staticFiles', 'uploads')
Session(app)


@app.route('/')
def index():
    ip = request.remote_addr
    return render_template('index.html',user_ip=ip)

@app.route('/explore')
def explore():
    return render_template('explore.html')

 # Define allowed files
ALLOWED_EXTENSIONS = {'csv'}
 
@app.route('/upload', methods=['GET', 'POST'])
def uploadFile():
    if request.method == 'POST':
      # upload file flask
        f = request.files.get('file')
 
        # Extracting uploaded file name
        data_filename = secure_filename(f.filename)
 
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],data_filename))
 
        session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'],data_filename)
         # Uploaded File Path
        data_file_path = session.get('uploaded_data_file_path', None)
        # read csv
        uploaded_df = pd.read_csv(data_file_path,encoding='unicode_escape')
        # Converting to html Table
        uploaded_df_html = uploaded_df.to_html()
        return render_template('view.html',
                           data_var=uploaded_df_html)
    return render_template("explore.html" )

 

@app.route('/algorithm-selection')
def display():
    ip = request.remote_addr
    # Uploaded File Path
    data_file_path = session.get('uploaded_data_file_path', None)
    current_app.logger.debug(f'Session data file path: {data_file_path}')
    
    if not data_file_path:
        current_app.logger.error('No uploaded data file path found in session')
        return jsonify({'message': 'No uploaded data file path found in session'}), 400

    if not os.path.exists(data_file_path):
        current_app.logger.error(f'File does not exist: {data_file_path}')
        return jsonify({'message': 'Uploaded data file path does not exist'}), 400

    try:
        # Read the CSV file
        df = pd.read_csv(data_file_path, encoding='unicode_escape')
    except Exception as e:
        app.logger.error(f'Error reading the CSV file: {e}')
        return jsonify({'message': f'Error reading the CSV file: {str(e)}'}), 500
    df_1 = pd.DataFrame()

    #Preprocessing of data

    #Finding the Relative Strength Index

    delta = df['Close'].diff(1)
    delta.dropna(inplace = True)
    positive = delta.copy()
    negative = delta.copy()

    positive[ positive < 0] = 0
    negative[ negative > 0] = 0

    days = 14

    average_gain = positive.rolling(window = days).mean()
    average_loss = abs(negative.rolling(window = days).mean())

    relative_strength = average_gain/average_loss

    RSI = 100.0 - (100.0/ (1.0 + relative_strength))
    df['RSI'] = RSI
    df['RSI'] = df['RSI'].fillna(0)

    #Finding Simple Moving Average
    period = 5
    SMA = df['Close'].rolling(window= period).mean()
    df['SMA'] = SMA
    df['SMA'] = df['SMA'].fillna(0)

    #Finding Exponential Moving Average
    EMA = df['Close'].ewm(span = 5, min_periods = 0, adjust = False).mean()
    df['EMA'] = EMA

    df['Date'] = pd.to_datetime(df['Date'], format= "mixed")

    df= df[['Date','Close','RSI', 'SMA', 'EMA']]

    df_1 = pd.DataFrame(df)
    csv_data = df_1.to_csv(index=False)
    session['csv_data'] = csv_data

    return render_template('display.html',user_ip=ip)

@app.route('/random-forest-regressor')
def rfmodel():
    data_file_path = session.get('uploaded_data_file_path', None)
    if data_file_path:
        return render_template('Random-forest-regressor.html')
    ip = request.remote_addr
    return render_template('display.html',user_ip=ip)


@app.route('/Gradient-boosting-regressor')
def gbmodel():
    data_file_path = session.get('uploaded_data_file_path', None)
    if data_file_path:
        return render_template('Gradient-boosting-regressor.html')

@app.route('/Lstm-model')
def lstmmodel():
    data_file_path = session.get('uploaded_data_file_path', None)
    if data_file_path:
        return render_template('Lstm-model.html')

@app.route('/Gru-model')
def grumodel():
    data_file_path = session.get('uploaded_data_file_path', None)
    if data_file_path:
        return render_template('Gru-model.html')




