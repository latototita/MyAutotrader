import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from metaapi_cloud_sdk import MetaApi
import asyncio,os
from sklearn.neural_network import MLPRegressor
# MetaApi API credentials
# Initialize MetaApi client
token = os.getenv('TOKEN') or 'eyJhbGciOiJSUzUxMiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2YjI0NTQ0ZWYzMWI0NzQ4NWMxNzQ1NmUzNzdmYTlhZiIsInBlcm1pc3Npb25zIjpbXSwidG9rZW5JZCI6IjIwMjEwMjEzIiwiaW1wZXJzb25hdGVkIjpmYWxzZSwicmVhbFVzZXJJZCI6IjZiMjQ1NDRlZjMxYjQ3NDg1YzE3NDU2ZTM3N2ZhOWFmIiwiaWF0IjoxNjgzOTcwNDk4fQ.XzOt-R6egTLGb0fpmbxzDrLHqqlSbqdskeX3OSbx585bi_jG9BhSp-PtEyZ4kqJBafXcGmGRa8IMYQ6BMtDRmoiUd6InEjioBhPlKa6wrylTruPK6_YYq3LsZGd-GctHqW5-_pv3UtKyYriHO-P61dE-zpH6AAAO-NeAru-GKvOQeNwhwSVW_Q8Ov6Q6dljt0q9psxZYOU2jZiR1N3d0d_pQpvKLCgXFk71TL93GyEj-7csQ5Z0py0ChVioeWY7Cf-MlzEJdnSFgcHeFaKfny680C-5srBJwCO4EBVSEEqJao71fhnnK7UsW_QVMUoamVEBvbxD2Wr0F2pHcdIkVUoMrJeNiWdCTvdEONsg9xMFREqGdvlx66khNhvOpVvK_obsSEwMUS7Qvk3-3yh5F7PaT0qsQW4WdZVRaTLbayA7ChbYqCGvp4EAA4mxYTSxWjihDFCWHy6QWmHVzDw5JzhUxus-bWtOTiVVGUjg5e5uPNSHYUzN2D0Pl4p6QxnGISQCmRTuNtbEEm_9yLF_5xuRAdQez1VS0rYP0x3YauLmLIdhpmNKjNNfi13uAiwJVmjIj__9VDALqiGje25WFWr9BLQCUZdemGHe4q9bc2IcAjSZIo6auI6aVqkGwm7UpkHat_FMTxynZnhDNkrGebjgvuW4_1nbmWVUGZ4y2mTc'
accountId = os.getenv('ACCOUNT_ID') or '44d6fa31-2cd5-4aaa-b5ed-8189b2d4a0b5'
symbol='XAUUSDm'
# Connect to MetaApi
timeframe='1m'
candlesn=1000000000

# Function to extract features from candlestick data
def extract_features(candles):
    features = []
    for i in range(len(candles)-1):
        candle = candles[i]
        next_candle = candles[i+1]
        features.append([
            candle['open'], candle['close'], candle['low'], candle['high'],
            next_candle['open'] - candle['close']
        ])
    return np.array(features)

# Function to convert labels to binary classes (0 or 1)
def convert_labels(labels):
    return [1 if label else 0 for label in labels]
def extract_labels(candles):
    labels = []
    for i in range(10, len(candles)):
        labels.append(candles[i]['close'])
    return np.array(labels)

# Function to generate buy/sell predictions and stop-loss/take-profit levels
def generate_predictions(model, features):
    predictions = model.predict(features)
    stop_loss = np.where(predictions == 1, features[:, 0] - (features[:, 3] - features[:, 2]), 0)
    take_profit = np.where(predictions == 0, features[:, 0] + (features[:, 3] - features[:, 2]), 0)
    return predictions, stop_loss, take_profit

async def main():
    api = MetaApi(token)
    account = await api.metatrader_account_api.get_account(accountId)
    initial_state = account.state
    deployed_states = ['DEPLOYING', 'DEPLOYED']
    if initial_state not in deployed_states:
        # Wait until account is deployed and connected to the broker
        print('Deploying account')
        await account.deploy()
    print('Waiting for API server to connect to the broker (may take a few minutes)')
    await account.wait_connected()

    # Connect to MetaApi API
    connection = account.get_rpc_connection()
    await connection.connect()

    # Wait until terminal state is synchronized to the local state
    print('Waiting for SDK to synchronize to terminal state (may take some time depending on your history size)')
    await connection.wait_synchronized()

    # Retrieve candlestick data asynchronously
    candles = await account.get_historical_candles(symbol=symbol, timeframe=timeframe, start_time=None, limit=candlesn)

    # Extract features and labels
    features = extract_features(candles)
    labels = extract_labels(candles)

    # Train the MLP regression model
    model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000)
    model.fit(features, labels)

    # Generate predictions for the next 10 minutes
    
    candless = await account.get_historical_candles(symbol='XAGUSDm', timeframe=timeframe, start_time=None, limit=candlesn)
    next=extract_features(candless)
    predictions = model.predict(next)

    # Print predicted prices for the next 10 minutes
    print('Predicted Prices for the Next 10 Minutes:')
    for i, prediction in enumerate(predictions):
        print(f'Time {i+1}: {prediction}')

# Run the main function asynchronously
asyncio.run(main())
