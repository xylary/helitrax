import string
import logging
import time
import sys
sys.path.append('/Users/danielknopf/github/tdameritrade')
import tdapi
from tdapi import TDAmeritradeAPI
import datetime
import keyring
import numpy as np
import pandas as pd
import json
import progressbar
import feature_extraction


logging.basicConfig(filename='fetch_equity_data.log',level=logging.INFO)

def get_price_history(td, tickers, days=108):
    price_dataframes = {}
    failed = []
    bar = progressbar.ProgressBar()
    for i in bar(range(len(tickers))):
        ticker = tickers[i]
        dt = datetime.datetime.now()
        first = True

        logging.info('%s: %s' % (ticker, dt.strftime('%Y%m%d')))
        try:
            df = td.getPriceHistory(ticker, intervalType='DAILY', intervalDuration='1', periodType='MONTH',
                period='3', startdate=None, enddate=dt.strftime('%Y%m%d'), extended=None)
        except ValueError:
            logging.info('Failed: %s' % ticker)
            failed.append(ticker)
            continue

        #fname = 'ohlcdata/%s_price_%s_%s.csv' % (ticker, fulldf.timestamp.min().date().strftime('%Y%m%d'), fulldf.timestamp.max().date().strftime('%Y%m%d'))
        #fulldf = fulldf.set_index(fulldf.timestamp, drop=False, inplace=False, verify_integrity=True).drop('timestamp',1)
        #fulldf.to_csv(fname)
        price_dataframes[ticker] = df


    logging.info('Failed tickers:')
    for ticker in failed:
        logging.info(ticker)

    return price_dataframes


def main():

    sequence_length = 20 # TODO - make this a flag
    batch_size = 200

    fp = open('tickers')
    lines = fp.readlines()[1:]
    fp.close()

    lines = [i.strip().split() for i in lines]
    tickers = [i[0] for i in lines]
    #logging.info(tickers)

    metadata = {}
    metadata['sequence_length'] = sequence_length
    metadata['batch_size'] = batch_size
    tickers = sorted(tickers)

    pw = keyring.get_password('tdameritrade','dgregknopf')
    td = TDAmeritradeAPI('GRNO')
    td.login('dgregknopf',pw)

    # Get price data
    dfs = get_price_history(td, tickers)

    # Extract features and drop initial columns
    timestamps = {}
    for key, df in dfs.items():
        df = feature_extraction.add_features(df, prune=True)
        df = df[-sequence_length:] # keep only the last set of observations
        ts = sorted(['%s' % v for v in df['timestamp'].values])
        df.drop('timestamp', axis=1, inplace=True)
        df.drop('open', axis=1, inplace=True)
        df.drop('high', axis=1, inplace=True)
        df.drop('low', axis=1, inplace=True)
        df.drop('close', axis=1, inplace=True)
        df.drop('volume', axis=1, inplace=True)
        dfs[key] = df
        timestamps[key] = ts
        #dfs[key].to_csv('newdata/%s.csv' % key)

    # Check for unexpected timestamps
    baseline = timestamps.values()[0]
    for key, val in timestamps.items():
        if set(val) != set(baseline):
            print('Warning!! Unexpected timestamps for ticker %s' % key)
    metadata['timestamps'] = baseline # indexed by ticker

    num_features = dfs.values()[0].shape[1]
    expected_shape = (sequence_length, num_features)

    metadata['num_features'] = num_features

    frames = []
    keys = sorted(dfs.keys())
    valid = []
    for key in keys:
        if dfs[key].values.shape == expected_shape:
            frames.append(dfs[key].values)
            valid.append(key)
        else:
            print('Unexpected shape: %s: %s' % (key, dfs[key].values.shape))
            #del timestamps[key]  # Don't really need the timestamps but might want for debug

    if (len(frames) % batch_size) != 0:
        pad = [np.zeros(frames[0].shape)] * (batch_size - (len(frames) % batch_size))
        #import pdb; pdb.set_trace()
        frames = np.concatenate((frames, pad))

    X_input = np.transpose(frames, [1,0,2])
    metadata['tickers'] = valid
    tag = datetime.datetime.now().strftime('%Y%m%d')
    npfilename = 'X_input_%s.data' % tag
    np.save(npfilename, X_input)
    metadata['numpy_datafile'] = npfilename
    with open('X_input_%s.metadata.json' % tag, 'w') as outfile:
        json.dump(metadata, outfile)

    #while len(frames) > 0:
    #    X_slice = frames[:batch_size]
    #    frames = frames[batch_size:]
    #    X_batch = np.transpose(X_slice, [1,0,2])

    #import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
