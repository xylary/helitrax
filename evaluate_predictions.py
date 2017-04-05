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
import tensorflow as tf
from dateutil.parser import parse

logging.basicConfig(filename='evaluate_predictions.log',level=logging.INFO)

tf.app.flags.DEFINE_string('inputfile', '', 'JSON input filename')
FLAGS = tf.app.flags.FLAGS

def get_price_history(td, tickers, days=108):
    price_dataframes = {}
    failed = []
    bar = progressbar.ProgressBar()
    for i in bar(range(len(tickers))):
        ticker = tickers[i]
        dt = datetime.datetime.now()
        first = True

        #logging.info('%s: %s' % (ticker, dt.strftime('%Y%m%d')))
        try:
            df = td.getPriceHistory(ticker, intervalType='DAILY', intervalDuration='1', periodType='MONTH',
                period='1', startdate=None, enddate=dt.strftime('%Y%m%d'), extended=None)
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

    if not FLAGS.inputfile:
        print('please specify input json file')
        return

    with open(FLAGS.inputfile) as f:
        inputdata = json.load(f)

    startdate = inputdata['startdate']
    predictions = inputdata['predictions']

    pw = keyring.get_password('tdameritrade','dgregknopf')
    td = TDAmeritradeAPI('GRNO')
    td.login('dgregknopf',pw)

    tickers = sorted(predictions.keys())
    # Get price data
    dfs = get_price_history(td, tickers)
    results = {}
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for key, df in dfs.items():
        #import pdb; pdb.set_trace()
        df = df.set_index(df['timestamp'])
        orig = df.ix[startdate].close
        delta = df.iloc[-1].close - orig
        pct = 100.0*delta/orig
        results[key] = delta
        s = ''
        if predictions[key] == 1:
            if pct >= 0:
                tp += 1
                s = 'tp'
            else:
                fp += 1
                s = 'fp'
        else:
            if pct <= 0:
                tn += 1
                s = 'tn'
            else:
                fn += 1
                s = 'fn'
        print('[%s] %5s: %d: %2f, %2f%%' % (s, key, predictions[key], delta, pct))
    print('Total scores: tp=%d, tn=%d, fp=%d, fn=%d' % (tp, tn, fp, fn))

if __name__ == '__main__':
    main()
