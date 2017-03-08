import pandas as pd
import numpy as np
import talib
import talib.abstract


#### OHLCV Functions ####
def closed_in_top_half_of_range(o,h,l,c,v):
    return int((c > (l + (h-l)/2.0)))

def closed_in_bottom_half_of_range(o,h,l,c,v):
    return int((c < (l + (h-l)/2.0)))

def lower_wick(o,h,l,c,v):
    return min(o,c)-l

def upper_wick(o,h,l,c,v):
    return h-max(o,c)

def full_range(o,h,l,c,v):
    return (h-l)

def real_body(o,h,l,c,v):
    return abs(c-o)

def range_vs_close(o,h,l,c,v):
    return full_range(o,h,l,c,v)/c

def body_vs_range(o,h,l,c,v):
    if full_range(o,h,l,c,v) == 0:
        return 0.0
    else:
        return real_body(o,h,l,c,v) / full_range(o,h,l,c,v)

def lower_wick_at_least_twice_real_body(o,h,l,c,v):
    return int(lower_wick(o,h,l,c,v) >= 2 * real_body(o,h,l,c,v))

def upper_wick_at_least_twice_real_body(o,h,l,c,v):
    return int(upper_wick(o,h,l,c,v) >= 2 * real_body(o,h,l,c,v))

def is_hammer(o,h,l,c,v):
    return lower_wick_at_least_twice_real_body(o,h,l,c,v) \
    and closed_in_top_half_of_range(o,h,l,c,v)

ohlcv_functions = [closed_in_top_half_of_range,
                   closed_in_bottom_half_of_range,
                   range_vs_close,
                   real_body,
                   body_vs_range,
                   lower_wick_at_least_twice_real_body,
                   upper_wick_at_least_twice_real_body,
                   is_hammer]
################

def add_features(df, prune=True):
    df['log_return'] = np.log(df['close']/df['close'].shift())
    df['log_volume'] = np.log(df['volume']/df['volume'].shift())

    for f in ohlcv_functions:
        df[f.__name__] = list(map(f, df["open"], df["high"], \
            df["low"], df["close"], df["volume"]))


    #################################
    # Add momentum indicators
    df['MFI'] = talib.abstract.Function('MFI')(df) / 100.0
    df['MFI_gt_80'] = (df['MFI'] > 0.8).astype(float)
    df['MFI_lt_20'] = (df['MFI'] < 0.2).astype(float)

    df['ADX'] = talib.abstract.Function('ADX')(df) / 100.0
    df['PLUS_DI'] = talib.abstract.Function('PLUS_DI')(df)
    df['MINUS_DI'] = talib.abstract.Function('MINUS_DI')(df)
    # ADX buy signal: PLUS_DI crosses above MINUS_DI and ADX > threshold
    adx_threshold = 0.25
    pos_di_cross = (df['PLUS_DI'] > df['MINUS_DI']).astype(int) * \
        (df['PLUS_DI'].shift() < df['MINUS_DI'].shift()).astype(int)
    df['ADX_BUY_SIGNAL'] = (df['ADX'] > adx_threshold).astype(int) * pos_di_cross
    neg_di_cross = (df['MINUS_DI'] > df['PLUS_DI']).astype(int) * (df['MINUS_DI'].shift() < df['PLUS_DI'].shift()).astype(int)
    df['ADX_SELL_SIGNAL'] = (df['ADX'] > adx_threshold).astype(int) * neg_di_cross
    df['ADX_TREND_STRENGTH'] = df['ADX'].apply(lambda x: x if x > adx_threshold else 0.0)
    df = df.drop('ADX', axis=1).drop('MINUS_DI', axis=1).drop('PLUS_DI', axis=1)

    # AROON
    func = talib.abstract.Function('AROON')
    df = pd.concat([df, func(df)], axis=1)
    df['AROONOSC'] = talib.abstract.Function('AROONOSC')(df) / 100.0
    df['aroonup'] = df['aroonup'] / 100.0
    df['aroondown'] = df['aroondown'] / 100.0
    df['AROON_SIGNAL'] = (df['aroonup'] > 0.70).astype(int) * (df['aroondown'] < 0.30).astype(int) - \
        (df['aroondown'] > 0.70).astype(int) * (df['aroonup'] < 0.30).astype(int)
    df = df.drop('aroonup', axis=1).drop('aroondown',axis=1)

    # Add pattern recognition indicators
    cdl_functions = talib.get_function_groups()['Pattern Recognition']
    for f in cdl_functions:
        func = talib.abstract.Function(f)
        tmp = func(df)
        df[f] = tmp/100.0

    if prune:
        # Drop any pattern recognition columns that are all zero's
        #zero_cdl_colummns = [col for col in df.columns if (df[col].max() == 0) and col in cdl_functions]
        #df = pd.concat([df.drop(col, axis=1) for col in zero_cdl_colummns])

        # Cleaner way -- this drops any column that is all zeros
        #df.loc[:, (df != 0).any(axis=0)]

        # Drop any rows with NaN (should only be at the beginning)
        df = df.dropna(axis=0)

    return df

def add_features_novolume(df, prune=True):
    df['log_return'] = np.log(df['close']/df['close'].shift())

    for f in ohlcv_functions:
        df[f.__name__] = list(map(f, df["open"], df["high"], \
            df["low"], df["close"], df["volume"]))

    df.drop('volume', axis=1)

    cdl_functions = talib.get_function_groups()['Pattern Recognition']
    for f in cdl_functions:
        func = talib.abstract.Function(f)
        tmp = func(df)
        df[f] = tmp/100.0

    if prune:
        # Drop any pattern recognition columns that are all zero's
        #zero_cdl_colummns = [col for col in df.columns if (df[col].max() == 0) and col in cdl_functions]
        #df = pd.concat([df.drop(col, axis=1) for col in zero_cdl_colummns])

        # Cleaner way -- this drops any column that is all zeros
        df.loc[:, (df != 0).any(axis=0)]

        # Drop any rows with NaN (should only be at the beginning)
        df = df.dropna(axis=0)

    return df
