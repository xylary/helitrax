import gridlstm
import datetime
import random
import string
import glob
import logging


logging.basicConfig(filename='log/orchestrator.log',level=logging.INFO)

hyperparams = {
    'num_classes': ['2'],
    'num_hidden': ['10', '50', '100', '150', '200', '500'],
    'num_layers': ['2', '3', '4'],
    'batch_size': ['50', '100', '200'],
    'sequence_length': ['25'],  # Note this must match the dataset
    #'num_epochs': ['1000', '3000', '5000'],
    'num_epochs': ['100'],
    'max_grad_norm': ['3.0', '5.0'],
    'pos_weight': ['0.01', '0.03', '0.1', '0.3', '1.0', '1.5', '2.0', '3.0'],
    'learning_rate': ['2e-3', '5e-3', '8e-3']
}

csvfiles = glob.glob("training_data2/*.csv")

results = {}
bestf1 = 0.0
for i in range(2):
    args = []
    for k in hyperparams.keys():
        args.append('--'+k)
        args.append(random.choice(hyperparams[k]))
    key = str.join(' ', args)
    if key in results.keys():
        continue
    else:
        args.append('--description')
        args.append('test%d' % i)
        logging.info('Running test %d: %s' % (i, args))
        for csv in csvfiles:
            args.append(csv)
        res = gridlstm.train_and_test(args)
        if res['test']['F1'] > bestf1:
            bestf1 = res['test']['F1']
            logging.info('New best F1 on test set:', bestf1)
        logging.info(res)
        results[key] = res
logging.info('Best F1 score:', bestf1)
