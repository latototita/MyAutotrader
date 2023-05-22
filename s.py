import asyncio
import numpy as np
import os
import pandas_ta as ta
from metaapi_cloud_sdk import MetaApi
import pandas as pd

# Initialize MetaApi client
token = os.getenv('TOKEN') or 'eyJhbGciOiJSUzUxMiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2YjI0NTQ0ZWYzMWI0NzQ4NWMxNzQ1NmUzNzdmYTlhZiIsInBlcm1pc3Npb25zIjpbXSwidG9rZW5JZCI6IjIwMjEwMjEzIiwiaW1wZXJzb25hdGVkIjpmYWxzZSwicmVhbFVzZXJJZCI6IjZiMjQ1NDRlZjMxYjQ3NDg1YzE3NDU2ZTM3N2ZhOWFmIiwiaWF0IjoxNjgzOTcwNDk4fQ.XzOt-R6egTLGb0fpmbxzDrLHqqlSbqdskeX3OSbx585bi_jG9BhSp-PtEyZ4kqJBafXcGmGRa8IMYQ6BMtDRmoiUd6InEjioBhPlKa6wrylTruPK6_YYq3LsZGd-GctHqW5-_pv3UtKyYriHO-P61dE-zpH6AAAO-NeAru-GKvOQeNwhwSVW_Q8Ov6Q6dljt0q9psxZYOU2jZiR1N3d0d_pQpvKLCgXFk71TL93GyEj-7csQ5Z0py0ChVioeWY7Cf-MlzEJdnSFgcHeFaKfny680C-5srBJwCO4EBVSEEqJao71fhnnK7UsW_QVMUoamVEBvbxD2Wr0F2pHcdIkVUoMrJeNiWdCTvdEONsg9xMFREqGdvlx66khNhvOpVvK_obsSEwMUS7Qvk3-3yh5F7PaT0qsQW4WdZVRaTLbayA7ChbYqCGvp4EAA4mxYTSxWjihDFCWHy6QWmHVzDw5JzhUxus-bWtOTiVVGUjg5e5uPNSHYUzN2D0Pl4p6QxnGISQCmRTuNtbEEm_9yLF_5xuRAdQez1VS0rYP0x3YauLmLIdhpmNKjNNfi13uAiwJVmjIj__9VDALqiGje25WFWr9BLQCUZdemGHe4q9bc2IcAjSZIo6auI6aVqkGwm7UpkHat_FMTxynZnhDNkrGebjgvuW4_1nbmWVUGZ4y2mTc'
accountId = os.getenv('ACCOUNT_ID') or '44d6fa31-2cd5-4aaa-b5ed-8189b2d4a0b5'

# Define parameters
symbol_list = ['XAUUSDm', 'GBPUSDm', 'XAGUSDm', 'AUDUSDm', 'EURUSDm', 'USDJPYm', 'GBPTRYm','AUDCADm','AUDCHFm','AUDJPYm','CADJPYm','CHFJPYm','EURCADm', 'EURAUDm','EURCHFm','EURGBPm','EURJPYm','GBPAUDm','GBPCADm', 'GBPCHFm','GBPJPYm','UK100m','HK50m','ADAUSDm','BATUSDm','BTCJPYm','BTCKRWm','BTCUSDm','DOTUSDm','ENJUSDm','FILUSDm','SNXUSDm']
rsi_period = 14
smi_period = 14
atr_period = 14
atr_multiplier = 3.0
bollinger_period = 20
bollinger_std = 2
ema_period = 20
adx_period = 14

def calculate_adx(high_prices, low_prices, close_prices, period=14):
    # Calculate True Range (TR)
    tr1 = high_prices - low_prices
    tr2 = np.abs(high_prices- np.roll(close_prices, 1))
    tr3 = np.abs(low_prices - np.roll(close_prices, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))

    # Calculate Directional Movement (+DM and -DM)
    up_move = high_prices- np.roll(high_prices, 1)
    down_move = np.roll(low_prices, 1) - low_prices
    up_move = np.where((up_move > 0) & (up_move > down_move), up_move, 0.0)
    up_move[(up_move <= 0) | (up_move <= down_move)] = 0.0
    minus_dm = np.where((down_move > 0) & (down_move > up_move), down_move, 0.0)
    minus_dm[(down_move <= 0) | (down_move <= up_move)] = 0.0

    plus_dm = np.where((up_move > 0) & (up_move > down_move), up_move, 0.0)


    # Smooth True Range and Directional Movement
    smoothed_tr = np.convolve(tr, np.ones(period), mode='valid') / period
    smoothed_plus_dm = np.convolve(plus_dm, np.ones(period), mode='valid') / period
    smoothed_minus_dm = np.convolve(minus_dm, np.ones(period), mode='valid') / period

    # Calculate +DI and -DIo
    plus_di = (smoothed_plus_dm / smoothed_tr) * 100
    minus_di = (smoothed_minus_dm / smoothed_tr) * 100

    # Calculate DX
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100

    # Smooth DX to calculate ADX
    adx = np.convolve(dx, np.ones(period), mode='valid') / period

    return adx

def calculate_ichimoku_cloud(high_prices, low_prices, conversion_period=9, base_period=26, leading_span_b_period=52, displacement=26):
    # Tenkan-sen (Conversion Line)
    conversion_line = (np.max(high_prices[-conversion_period:]) + np.min(low_prices[-conversion_period:])) / 2

    # Kijun-sen (Base Line)
    base_line = (np.max(high_prices[-base_period:]) + np.min(low_prices[-base_period:])) / 2

    # Senkou Span A (Leading Span A)
    leading_span_a = (conversion_line + base_line) / 2

    # Senkou Span B (Leading Span B)
    leading_span_b = (np.max(high_prices[-leading_span_b_period:]) + np.min(low_prices[-leading_span_b_period:])) / 2

    # Shift the Leading Span A and Leading Span B values forward
    leading_span_a = np.roll(leading_span_a, displacement)
    leading_span_b = np.roll(leading_span_b, displacement)

    return conversion_line, base_line, leading_span_a, leading_span_b

def ichimoku_cloud_strategy(high_prices, low_prices, close_prices):
    conversion_line, base_line, leading_span_a, leading_span_b = calculate_ichimoku_cloud(high_prices, low_prices)

    current_close = close_prices[-1]
    previous_close = close_prices[-2]

    # Check for trend direction
    if current_close > leading_span_a and current_close > leading_span_b:
        trend_direction = "Strong Uptrend"
    elif current_close < leading_span_a and current_close < leading_span_b:
        trend_direction = "Strong Downtrend"
    elif current_close > leading_span_a or current_close > leading_span_b:
        trend_direction = "Weak Uptrend"
    else:
        trend_direction = "Weak Downtrend"

    # Trading signals based on trend direction
    if trend_direction == "Strong Uptrend":
        if previous_close < conversion_line and previous_close < base_line:
            return "Buy Signal"

    if trend_direction == "Strong Downtrend":
        if previous_close > conversion_line and previous_close > base_line:
            return "Sell Signal"

    return "No Signal"

async def main():
    # Connect to the MetaTrader account
    while True:
        api = MetaApi(token)
        account = await api.metatrader_account_api.get_account(accountId)
        initial_state = account.state
        deployed_states = ['DEPLOYING', 'DEPLOYED']
        if initial_state not in deployed_states:
                # wait until account is deployed and connected to broker
                print('Deploying account')
                await account.deploy()
        print('Waiting for API server to connect to broker (may take a few minutes)')
        await account.wait_connected()
        
        # connect to MetaApi API
        connection = account.get_rpc_connection()
        await connection.connect()
        
        # wait until terminal state synchronized to the local state
        print('Waiting for SDK to synchronize to terminal state (may take some time depending on your history size)')
        await connection.wait_synchronized()
        while True:
            # Check for open trades
            trades = await connection.get_orders()
            if trades:
                print("There are open trades. Skipping analysis.")
            else:
                for symbol in symbol_list:
                    try:
                        # Fetch historical price data
                        candles = await account.get_historical_candles(symbol=symbol, timeframe='1m', start_time=None, limit=1000000)
                        print('Fetched the latest candle data successfully')
                        
                        # Convert candles to DataFrame
                        df = pd.DataFrame(candles)
                        df.set_index('time', inplace=True)
                        df['close'] = df['close'].astype(float)
                    except Exception as e:
                        print(f"Error retrieving candle data: {e}")
                        continue

                    # Apply pandas_ta indicators
                    close_pricesb = df['close'].values
                    high_prices = np.array([candle['high'] for candle in candles])
                    low_prices = np.array([candle['low'] for candle in candles])
                    close_prices = np.array([candle['close'] for candle in candles])
                    ichimoku = ichimoku_cloud_strategy(high_prices, low_prices, close_prices)
                    # smi number 1
                    df['smi_ema'] = ta.ema(df['close'], length=smi_period)
                    # rsi number 2
                    df['rsi'] = ta.rsi(df['close'], length=rsi_period)
                    # bbb number 3
                    df['bollinger_middle'] = ta.sma(df['close'], length=bollinger_period)
                    df['bollinger_std'] = ta.stdev(df['close'], length=bollinger_period)
                    df['bollinger_upper'] = df['bollinger_middle'] + bollinger_std * df['bollinger_std']
                    df['bollinger_lower'] = df['bollinger_middle'] - bollinger_std * df['bollinger_std']
                    # ema number 4
                    df['ema'] = ta.ema(df['close'], length=ema_period)
                    # adx number 5
                    adx = calculate_adx(high_prices, low_prices, close_prices)
                    # Perform trading logic using Bollinger Bands
                    upper_band = df['bollinger_upper'].values
                    lower_band = df['bollinger_lower'].values
                    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=atr_period)
                    atr = df['atr'].to_numpy()
                    print('B n S teating started')
                    buy_signal = (
                        df['close'][-1] > df['smi_ema'][-1]
                    ) and (
                        df['rsi'][-1] >50
                    ) and (
                        ichimoku =="Buy Signal"
                    ) and(
                        close_pricesb[-1] > upper_band[-1] and 
                        close_pricesb[-2] < upper_band[-2]
                    ) and (
                        close_prices[-1] > df['ema'][-1]
                    )and (
                        adx[-1]> 25)
                        
                        
                    sell_signal = (
                        df['close'][-1] < df['smi_ema'][-1]
                    ) and (
                        df['rsi'][-1] <50
                    ) and (
                        ichimoku=="Sell Signal"
                    ) and (
                        close_prices[-1] < lower_band[-1]
                        and
                        close_prices[-2] > lower_band[-2]
                    ) and (
                        close_prices[-1] < df['ema'][-1]
                    ) and (
                        adx[-1]> 25)
                    print('B n S teated')
                    # Execute trading orders
                    prices = await connection.get_symbol_price(symbol)
                    current_price = prices['ask']
                    if buy_signal==True:
                        # Calculate prices at pips above and below the current price
                        print('take profit starting')
                        take_profit = current_price + (atr[-1] * float(5.0))
                        stop_loss = current_price - (atr[-1] * atr_multiplier)
                        print('calculations ended')
                        try:
                            # calculate margin required for trade
                            first_margin= await connection.calculate_margin({
                                'symbol': symbol,
                                'type': 'ORDER_TYPE_BUY',
                                'volume': 0.01,
                                'openPrice':  current_price
                            })
                            
                            first_margin=float(first_margin['margin'])
                            
                            if first_margin<float(0.8):
                                result = await connection.create_market_buy_order(
                                    symbol,
                                    0.01,
                                    stop_loss,
                                    take_profit,
                                    {'trailingStopLoss': {
                                            'distance': {
                                                'distance': 3,
                                                'units':'RELATIVE_PIPS'
                                            }
                                        }
                                    })
                            else:
                                pass
                            print('Trade successful, result code is ' + result['stringCode'])
                            await asyncio.sleep(240)
                        except Exception as err:
                            print('Buy Trade failed with error:')
                            print(api.format_error(err))
                    if sell_signal==True:
                        # Calculate prices at pips above and below the current price
                        take_profit = current_price - (atr[-1] * float(5.0))
                        stop_loss = current_price + (atr[-1] * atr_multiplier)
                        try:
                            # calculate margin required for trade
                            
                            first_margin= await connection.calculate_margin({
                                'symbol': symbol,
                                'type': 'ORDER_TYPE_SELL',
                                'volume': 0.01,
                                'openPrice':  current_price,
                            })
                            
                            first_margin=float(first_margin['margin'])
                            
                            if first_margin<float(0.8):
                                    result = await connection.create_market_sell_order(
                                    symbol,
                                    0.01,
                                    stop_loss,
                                    take_profit,
                                    {'trailingStopLoss': {
                                            'distance': {
                                                'distance': 3,
                                                'units':'RELATIVE_PIPS'
                                            }
                                        }
                                    })
                            else:
                                pass
                            print('Sell trade placed successfully')
                            await asyncio.sleep(240)
                        except Exception as err:
                            print('Trade failed with error:')
                            print(api.format_error(err))
                            
                    trades = await connection.get_orders()
                    if trades:
                        print("There are open trades. Skipping analysis.")
                        break
                    
            await asyncio.sleep(60)  # Sleep for 1 minute before the next iteration
        await asyncio.sleep(60)  # Sleep for 1 minute before the next iteration
asyncio.run(main())
