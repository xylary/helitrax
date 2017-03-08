import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import grid_rnn
import argparse
import logging
import os
import os.path
import datetime
import math
import random

logger = logging.getLogger(__name__)

class BatchDataHelper:
    def __init__(self, csvfiles, batch_size, sequence_length, train_pct=0.70, num_classes=2,
        max_csvfiles=20):
        self._df = {'train': {},
                    'test': {}}

        self.csvfiles = csvfiles
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.num_features = 0

        assert(len(csvfiles) > 0), logger.error('Empty list of csvfiles')

        self._ptr = {'train': (self.csvfiles[0], 0),
                     'test':  (self.csvfiles[0], 0)}

        y_pos_counts = {}
        logger.debug('Loading %d dataframes' % len(csvfiles))
        for csv in csvfiles:
            df = pd.read_csv(csv, index_col=0)

            # Truncate to batch size
            # TODO - remove silent truncation -- perhaps add a warning and
            # do this during the data prep pipeline
            chop = df.shape[0] % (batch_size + sequence_length)
            df = df[:-chop]
            y_pos_counts[csv] = df['ypos'].sum()

            # Split into train & test. Note - these are different time series so
            # we need to keep them separate. We will batch them in order.
            training_set_size = int((int(len(df) * train_pct) / batch_size) * batch_size)
            #logger.info('Training set size: %d' % training_set_size)
            self._df['train'][csv] = df[:training_set_size]
            self._df['test'][csv] = df[training_set_size:]
            logger.info('%s: train[%d] test[%d]' % (csv, len(self._df['train'][csv]),
                len(self._df['test'][csv])))
            if self._df['test'][csv]['ypos'].sum() == 0:
                logger.error('No positives in test set!! %s' % csv)

        # Truncate csvfiles to the subset with the most positive examples
        # NOTE - this is probably not good practice in general since it biases
        # the model towards series that were trending up, e.g. less sparse in this case
        # TODO - clean this up later as it's a little sloppy anyway
        if max_csvfiles < len(csvfiles):
            sortedcsvs = sorted(y_pos_counts, key=y_pos_counts.get, reverse=True)
            for i in range(len(sortedcsvs)):
                csv = sortedcsvs[i]
                if i < max_csvfiles:
                    logger.debug('%s: %d' % (csv, y_pos_counts[csv]))
                else:
                    del self._df['train'][csv]
                    del self._df['test'][csv]
            self.csvfiles = sortedcsvs[:max_csvfiles]
            logger.info('Truncated to the following %d csvs: %s' % (max_csvfiles, self.csvfiles))

        #for csv in self.csvfiles:
        #    logger.info('%s: train[%d] test[%d]' % (csv, len(self._df['train'][csv]),
        #        len(self._df['test'][csv])))
        self.num_features = len(df.columns) - num_classes


    def rewind(self):
        self._ptr = {'train': (self.csvfiles[0], 0),
                     'test':  (self.csvfiles[0], 0)}


    def next_batch(self, which='train', repeat=False, verbose=False, rotate=False):
        """Returns X_batch, y_batch or None if finished. If repeat=True,
        will continue returning data."""

        # Generate a batch by applying a sliding window to the data until
        # we have filled the batch_size.
        csv, offset = self._ptr[which]
        if not csv:
            return None, None
        #pdb.set_trace()

        #if csv == 'training_data/training_set_std_ABT.csv' and offset == 1400:
        #    pdb.set_trace()

        X = self._df[which][csv][self._df[which][csv].columns[self.num_classes:]]
        y = self._df[which][csv][self._df[which][csv].columns[:self.num_classes]]

        X_batch = []
        i = self.batch_size
        y_batch = y[offset+self.sequence_length-1:offset+self.sequence_length-1+i].values

        # Collect sliding windows of length sequence_length
        for i in range(self.batch_size):
            X_batch.append(X[offset+i:offset+i+self.sequence_length].values)


        if rotate:
            # Advance to next csvfile
            if csv == self.csvfiles[-1]:
                csv = self.csvfiles[0]
                offset += self.batch_size
            else:
                csv = self.csvfiles[self.csvfiles.index(csv) + 1]
            if offset + self.batch_size + self.sequence_length > self._df[which][csv].shape[0]:
                if repeat:
                    offset = 0
                else:
                    csv = None

        else: # Don't rotate
            offset += self.batch_size
            if offset + self.batch_size + self.sequence_length > self._df[which][csv].shape[0]:
                # We've reached the end, so advance to next csvfile. If this is the last one, check whether
                # or not we should repeat. In any case, reset the offset.
                offset = 0
                if csv == self.csvfiles[-1]:
                    if repeat:
                        csv = self.csvfiles[0]
                    else:
                        csv = None
                else:
                    csv = self.csvfiles[self.csvfiles.index(csv) + 1]

        self._ptr[which] = (csv, offset)
        if verbose:
            logger.info('Batch: %s, %d' % (csv, offset))

        # Should never hit either of these cases
        if len(X_batch) != self.batch_size:
            raise ValueError('X_batch length = %d' % len(X_batch))
        if len(y_batch) != self.batch_size:
            raise ValueError('y_batch length = %d' % len(y_batch))

        # Validate that one of the y values is nonzero in every cases
        for row in range(len(y_batch)):
            if y_batch[row][0] == 0 and y_batch[row][1] == 0:
                raise ValueError('y_batch has no valid class @ row %d' % row)

        return X_batch, y_batch

def evaluate_performance(model, actual_classes, session, feed_dict):
    predictions = tf.argmax(model, 1)
    actuals = tf.argmax(actual_classes, 1)

    ones_like_actuals = tf.ones_like(actuals)
    zeros_like_actuals = tf.zeros_like(actuals)
    ones_like_predictions = tf.ones_like(predictions)
    zeros_like_predictions = tf.zeros_like(predictions)

    tp_op = tf.reduce_sum(
    tf.cast(
      tf.logical_and(
        tf.equal(actuals, ones_like_actuals),
        tf.equal(predictions, ones_like_predictions)
      ),
      "float"
    )
    )

    tn_op = tf.reduce_sum(
    tf.cast(
      tf.logical_and(
        tf.equal(actuals, zeros_like_actuals),
        tf.equal(predictions, zeros_like_predictions)
      ),
      "float"
    )
    )

    fp_op = tf.reduce_sum(
    tf.cast(
      tf.logical_and(
        tf.equal(actuals, zeros_like_actuals),
        tf.equal(predictions, ones_like_predictions)
      ),
      "float"
    )
    )

    fn_op = tf.reduce_sum(
    tf.cast(
      tf.logical_and(
        tf.equal(actuals, ones_like_actuals),
        tf.equal(predictions, zeros_like_predictions)
      ),
      "float"
    )
    )

    tp, tn, fp, fn = \
    session.run(
      [tp_op, tn_op, fp_op, fn_op],
      feed_dict
    )
    return (tp, tn, fp, fn)

def performance_metrics(tp, tn, fp, fn):
    logger.info('Positives: true=%d false=%d' % (int(tp), int(fp)))
    logger.info('Negatives: true=%d false=%d' % (int(tn), int(fn)))

    # preload defaults
    metrics = {}
    metrics['accuracy'] = 0
    metrics['precision'] = 0
    metrics['F1'] = np.NaN
    metrics['recall'] = 0

    if (tp+fp == 0) or (tp+fn == 0):
        logger.info('No positives')
        metrics['error'] = 'No positives'
        return metrics

    tpr = float(tp)/(float(tp) + float(fn))
    fpr = float(fp)/(float(tp) + float(fn))

    accuracy = (float(tp) + float(tn))/(float(tp) + float(fp) + float(fn) + float(tn))
    recall = tpr

    precision = float(tp)/(float(tp) + float(fp))
    logger.info('Precision = %f' % precision)
    logger.info('Recall = %f' % recall)

    metrics['precision'] = precision
    metrics['recall'] = recall
    if (precision + recall) == 0:
        logger.info('F1 Score = Nan')
        metrics['F1'] = 'NaN'
    else:
        f1_score = (2 * (precision * recall)) / (precision + recall)
        logger.info('F1 Score = %f' % f1_score)
        metrics['F1'] = float(f1_score)
    logger.info('Accuracy = %f' % accuracy)
    metrics['accuracy'] = accuracy
    return metrics

def train_and_test(arguments):

    # Load arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--description', type=str, default='%s' % datetime.datetime.now().isoformat(),
                        help='descriptive name for this particular run')
    parser.add_argument('--output_dir', type=str, default='log',
                        help='directory to store logs and models')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='number of classes in the dataset - note these will be the first columns')
    parser.add_argument('--num_hidden', type=int, default=100,
                        help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in the RNN')
    parser.add_argument('--batch_size', type=int, default=200,
                        help='minibatch size')
    parser.add_argument('--sequence_length', type=int, default=20,
                        help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=5000,
                        help='number of epochs')
    parser.add_argument('--max_grad_norm', type=float, default=5.,
                        help='clip gradients at this value')
    parser.add_argument('--pos_weight', type=float, default=0.03,
                        help='weight for positive classification in cross-entropy loss')
    parser.add_argument('--learning_rate', type=float, default=5e-3,
                        help='eta passed to optimizer constructor')
    parser.add_argument('--fake', dest='fake', action='store_true', default=False)
    parser.add_argument('csvfiles', nargs='+',
                        help='csv file(s) containing training data')
    args = parser.parse_args(arguments)

    # Fake mode for debugging purposes
    if args.fake:
        results = [
        #1
        {'test': {'precision': random.random(), 'accuracy': random.random(), 'tn': 388.0, 'fn': 4.0,
        'recall': 0.0, 'F1': random.random(), 'tp': 0.0, 'fp': 8.0},
        'train': {'precision': random.random(), 'accuracy': random.random(),
        'tn': 1061.0, 'fn': 29.0, 'recall': random.random(), 'F1': random.random(), 'tp': 100.0, 'fp': 10.0}},
        #2
        {'train': {'tn': 2191.0, 'recall': 0.5270935960591133, 'fp': 6.0, 'accuracy': 0.9575, 'precision': 0.9469026548672567,
        'F1': 0.6772151898734177, 'tp': 107.0, 'fn': 96.0}, 'test': {'tn': 939.0, 'recall': 0.03125,
        'fp': 29.0, 'accuracy': 0.94, 'precision': 0.03333333333333333, 'F1': 0.03225806451612904, 'tp': 1.0, 'fn': 31.0}},
        #3
        {'train': {'tn': 2196.0, 'recall': 0.9753694581280788, 'fp': 1.0, 'accuracy': 0.9975, 'precision': 0.9949748743718593,
        'F1': 0.9850746268656716, 'tp': 198.0, 'fn': 5.0}, 'test': {'tn': 933.0, 'recall': 0.0, 'fp': 40.0,
        'accuracy': 0.933, 'precision': 0.0, 'F1': 'NaN', 'tp': 0.0, 'fn': 27.0}}
        ]
        return random.choice(results)

    # Make log dir if needed
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    fh = logging.FileHandler(os.path.join(args.output_dir, 'gridlstm_%s.log' % args.description.replace(' ','_')))
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)

    for k, v in vars(args).items():
        if k != 'csvfiles':
            logger.debug('%s: %s' % (k, v))
    logger.debug('csv files:')
    for csv in args.csvfiles:
        logger.debug(csv)
    logger.debug('end of csv files')

    # Load data
    csvfiles = args.csvfiles
    batch_size = args.batch_size
    num_classes = args.num_classes
    sequence_length = args.sequence_length
    helper = BatchDataHelper(csvfiles, batch_size, sequence_length, num_classes=num_classes)
    num_features = helper.num_features


    ###################
    # Construct the RNN
    tf.reset_default_graph()
    # Tensorflow requires input as a tensor (a Tensorflow variable) of the
    # dimensions [batch_size, sequence_length, input_dimension] (a 3d variable).
    # [Batch Size, Sequence Length, Input Dimension]. We let the batch size be unknown and to be determined at runtime
    with tf.name_scope('Inputs') as scope:
        feature_data = tf.placeholder(tf.float32, [args.batch_size, args.sequence_length, num_features])
        actual_classes = tf.placeholder(tf.float32, [args.batch_size, num_classes])

    # Create Grid2LSTM
    lstm_cell = grid_rnn.Grid2LSTMCell(args.num_hidden)  ## Ok, grid doesn't seem to work with dynamic_rnn

    # inputs: input Tensor, 2D, batch x input_size. Or None
    #      state: state Tensor, 2D, batch x state_size. Note that state_size =
    #        cell_state_size * recurrent_dims
    #      scope: VariableScope for the created subgraph; defaults to "GridRNNCell".
    logger.debug('LSTM input size: %s' % lstm_cell.input_size)
    logger.debug('LSTM output size: %s' % lstm_cell.output_size)
    logger.debug('LSTM state size: %s' % lstm_cell.state_size)
    lstm_output_size = lstm_cell.output_size

    # Create multiple layers
    lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * args.num_layers)

    initial_state = state = lstm_cell.zero_state(batch_size, tf.float32)
    outputs = []

    with tf.variable_scope('LSTM_Grid') as scope:
        for i in range(sequence_length):
            if i > 0: tf.get_variable_scope().reuse_variables()
            output, state = lstm_cell(feature_data[:, i, :], state)
        last_output = output

    # Feed the output of the LSTM into a softmax layer
    with tf.name_scope('Softmax') as scope:
        weights = tf.Variable(tf.truncated_normal([lstm_output_size, num_classes], stddev=0.01))
        biases = tf.Variable(tf.ones([num_classes]))
        logits = tf.matmul(last_output, weights) + biases
        model = tf.nn.softmax(logits)

    with tf.name_scope('Cost_Evaluation') as scope:
        # Use weighted cross entropy cost function since our dataset is relatively sparse.
        # We need to be able to dial in the penalty of missing a positive classification
        # so that the model doesn't always predict negatives.
        loss = tf.nn.weighted_cross_entropy_with_logits(logits, actual_classes, args.pos_weight, name='Weighted_cross_entropy')
        cost = tf.reduce_sum(loss) / batch_size
        #cost = tf.reduce_sum(loss)

    # Dimensions (assuming num_classes=2, hidden_size=24)
    # actual_classes has shape [batch_size, 2] where the 2 y columns are y_pos and y_neg
    # weights: [24, 2], seeded by random distribution
    # biases [2]
    # what is shape of model?
    # output*weights is [batch_size x 24] X [24, 2] --> [batch_size x 2]
    with tf.name_scope('Optimizer') as scope:
        # Create an Adam optimizer and connect it to the cost function node
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),args.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(args.learning_rate)
        gradients = zip(grads, tvars)
        training_step = optimizer.apply_gradients(gradients)

    with tf.name_scope('Accuracy_Evaluation') as scope:
        # Add ops to evaluate accuracy
        correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(actual_classes, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        cost_summary = tf.scalar_summary("cost", cost)
        training_summary = tf.scalar_summary("training_accuracy", accuracy)

    # Initialize the session
    session = tf.Session()
    init = tf.global_variables_initializer()
    session.run(init)

    writer = tf.train.SummaryWriter(os.path.join(args.output_dir, args.description), session.graph)

    # Train the model
    logger.info('Training model...')
    helper.rewind()
    for i in range(args.num_epochs):
        # TODO - figure out why the tensorboard is showing that we stop at ~800 samples. Is repeat not working?
        X_batch, y_batch = helper.next_batch('train', repeat=True, verbose=False, rotate=False)
        session.run(
            training_step,
            feed_dict={
              feature_data: X_batch,
              actual_classes: y_batch
        })
        if i%200 == 1:
            # Log cost function to TensorBoard
            cost_summ, training_summ = session.run(
                [cost_summary, training_summary],
                feed_dict={
                  feature_data: X_batch,
                  actual_classes: y_batch
            })
            writer.add_summary(cost_summ, i)
            writer.add_summary(training_summ, i)

        del X_batch
        del y_batch

    # Evaluate performance over the entire training set
    logger.info('Evaluating model performance on training set...')
    scores = [0,0,0,0]
    helper.rewind()
    results = {
        'train': {},
        'test': {}
    }
    while True:
        X_batch, y_batch = helper.next_batch('train', repeat=False, verbose=False, rotate=False)
        if (X_batch is None) or (y_batch is None):
            break

        tp, tn, fp, fn = evaluate_performance(model, y_batch, session, feed_dict={
              feature_data: X_batch,
              actual_classes: y_batch
        })
        scores[0] += tp; scores[1] += tn; scores[2] += fp; scores[3] += fn
        del X_batch
        del y_batch
    mets = performance_metrics(*scores)
    results['train']['tp'] = scores[0]
    results['train']['tn'] = scores[1]
    results['train']['fp'] = scores[2]
    results['train']['fn'] = scores[3]
    for k in mets.keys():
        results['train'][k] = mets[k]

    # Evaluate performance over the entire test set
    logger.info('Evaluating model performance on test set...')
    scores = [0,0,0,0]
    helper.rewind()
    while True:
        X_batch, y_batch = helper.next_batch('test', repeat=False, verbose=False, rotate=False)
        if (X_batch is None) or (y_batch is None):
            break

        tp, tn, fp, fn = evaluate_performance(model, y_batch, session, feed_dict={
              feature_data: X_batch,
              actual_classes: y_batch
        })
        scores[0] += tp; scores[1] += tn; scores[2] += fp; scores[3] += fn
        #print(scores)
        del X_batch
        del y_batch
    mets = performance_metrics(*scores)
    results['test']['tp'] = scores[0]
    results['test']['tn'] = scores[1]
    results['test']['fp'] = scores[2]
    results['test']['fn'] = scores[3]
    for k in mets.keys():
        results['test'][k] = mets[k]

    session.close()
    del helper

    return results

def main(args):
    train_and_test(args)

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
