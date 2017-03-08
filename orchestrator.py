import gridlstm
import datetime
import random
import string
import glob
import logging
import pandas as pd
import numpy as np


logging.basicConfig(filename='log/orchestrator.log',level=logging.INFO)

hyperparams_multiple = {
    'num_classes': ['2'],
    'num_hidden': ['10', '50', '100', '150', '200', '500'],
    'num_layers': ['2'],
    'batch_size': ['50', '100', '200'],
    'sequence_length': ['20'],  # Note this must match the dataset
    'num_epochs': ['1000', '3000'],
    'max_grad_norm': ['3.0', '5.0'],
    'pos_weight': ['0.3', '1.0', '2.0'],
    'learning_rate': ['2e-3', '5e-3', '2e-2']
}

hyperparams_single = {
    'num_classes': ['2'],
    'num_hidden': ['200'],
    'num_layers': ['2'],
    'batch_size': ['200'],
    'sequence_length': ['20'],  # Note this must match the dataset
    'num_epochs': ['1000'],
    'max_grad_norm': ['5.0'],
    'pos_weight': ['0.03'],
    'learning_rate': ['5e-3']
}


class Leaderboard:
    def __init__(self, columns=None):
        self.rows = []
        if columns:
            self.columns = columns
        else:
            self.columns = ['id', 'test_error', 'train_error',
                'F1_test', 'F1_train',
                'tp_test', 'tn_test', 'fp_test', 'fn_test',
                'precision_test','accuracy_test','recall_test',
                'tp_train', 'tn_train','fp_train','fn_train',
                'precision_train','accuracy_train','recall_train']

    def update(self, args, results, sortby='F1_test'):
        # Flatten the results dictionary
        newresults = {}
        newresults['args'] = str.join(' ', args)
        for k in results.keys():
            for key, val in results[k].items():
                newresults[key+'_'+k] = val
        self.rows.append(newresults)
        # Capture any new columns
        for col in newresults.keys():
            if col not in self.columns:
                self.columns.append(col)

        # Write out as csv
        df = pd.DataFrame(index=np.arange(len(self.rows)), data=self.rows,
            columns=self.columns)
        df.to_csv('log/leaderboard_nonsorted.csv')
        #df.sort_values(sortby, axis=0, ascending=False,
        #    na_position='last').to_csv('log/leaderboard.csv')

def build_args(hyperparams, csvfiles, description, randomize=False, quickTest=False, fakeMode=False):
    args = []
    if quickTest:
        hyperparams['num_epochs'] = ['10']

    for k in hyperparams.keys():
        args.append('--'+k)
        if randomize:
            args.append(random.choice(hyperparams[k]))
        else:
            args.append(hyperparams[k][0])
    if fakeMode:
        # Fake for quick debugging
        args.append('--fake')

    for csv in csvfiles:
        args.append(csv)

    return args

def main():
    # Set knobs here
    #csvfiles = glob.glob("training_data2/AAPL.csv")
    csvfiles = glob.glob("training_data2/*.csv")
    randomize = True
    quick = False
    fake = False
    results = {}

    if randomize:
        perms = 1
        for k, v in hyperparams_multiple.items():
            perms *= len(v)
        testrange = range(perms*2)
    else:
        testrange = range(len(csvfiles))


    lb = Leaderboard()
    for i in testrange:
        if randomize:
            csv = random.choice(csvfiles)  # randomly choose a csv. (List may have only 1 entry)
            ticker = csv.split('/')[1].split('.')[0]
            args = build_args(hyperparams_multiple, [csv], '%s_%d' % (ticker, i), randomize=True,
                quickTest=quick, fakeMode=fake)

        else:
            csv = csvfiles[i] # process csvs one at a time
            ticker = csv.split('/')[1].split('.')[0]
            args = build_args(hyperparams_multiple, [csv], '%s_%d' % (ticker, i), randomize=False,
                quickTest=quick, fakeMode=fake)


        key = str.join(' ', args)
        if key in results.keys():
            continue

        logging.info('Running test %d: %s: %s' % (i, csv, args))
        # Construct a model, train it, and test it with the given args
        try:
            res = gridlstm.train_and_test(args)
        except OSError:
            logging.error('Failed on args: %s' % args)
            continue

        logging.info(res)
        results[key] = res
        logging.info('Results test %d: %s: %s' % (i, csv, res))
        lb.update(args, res)


main()
