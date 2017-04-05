import sys
import threading
from grpc.beta import implementations
import numpy as np
import glob
import logging
import json
import progressbar
import tensorflow as tf
import tensorflow.contrib.util as util
sys.path.append('/serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server_test_client.runfiles/tf_serving/')
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow.python.framework import dtypes
import gridlstm
from gridlstm import BatchDataHelper

# Create a logger
logger = logging.getLogger(__name__)


# Define app arguments
tf.app.flags.DEFINE_integer('concurrency', 1,
                            'maximum number of concurrent inference requests')
tf.app.flags.DEFINE_integer('batch_size', 200,
                            'batch size model expects')
tf.app.flags.DEFINE_integer('sequence_length', 20,
                            'sequence length model expects')
tf.app.flags.DEFINE_string('server', '', 'PredictionService host:port')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory. ')
tf.app.flags.DEFINE_string('command', 'inference', 'Command to run. Valid commands: inference, backtest, test')

tf.app.flags.DEFINE_string('inputfile', '', 'JSON input filename')

FLAGS = tf.app.flags.FLAGS

def do_inference(hostport, work_dir, concurrency, X, batch_size=200, y=None):
    """Tests PredictionService with concurrent requests.
    Args:
    hostport: Host:port address of the PredictionService.
    work_dir: The full path of working directory for test data set.
    concurrency: Maximum number of concurrent requests.
    num_tests: Number of test images to use.
    Returns:
    The classification error rate.
    Raises:
    IOError: An error occurred processing test data set.
    """
    host, port = hostport.split(':')
    channel = implementations.insecure_channel(host, int(port))
    #stub = prediction_service_pb2.PredictionServiceStub(channel)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'helitrax' # Note this is associated with --model_name argument to tensorflow_model_server
    #request.model_spec.signature_name = signature_constants.CLASSIFY_INPUTS
    request.model_spec.signature_name = 'classify'


    #import pdb; pdb.set_trace()
    num_batches = X.shape[1] / batch_size
    X_batches = np.hsplit(X, num_batches)
    check = False
    if not (y is None):
        check = True
        y_batches = np.hsplit(y, num_batches)

    results = []
    scores = [0,0,0,0] # tp, tn, fp, fn
    for i in range(len(X_batches)):
        X = X_batches[i]
        request.inputs[signature_constants.CLASSIFY_INPUTS].CopyFrom(
            tf.contrib.util.make_tensor_proto(X, dtype=dtypes.float32, shape=[20, 200, 79]))
        result = stub.Predict(request, 100.0)
        ary = util.make_ndarray(result.outputs['classes'])
        results.append(ary)

        if check:
            predictions = ary
            for j in range(len(y_batches[i])):
                if y_batches[i][j][1] == 1:
                    if predictions[i] == 1:
                        scores[0] += 1
                    else:
                        scores[3] += 1
                if y_batches[i][j][1] == 0:
                    if predictions[i] == 1:
                        scores[2] += 1
                    else:
                        scores[1] += 1

    return (np.concatenate(results), scores)

def do_test():
    helper = BatchDataHelper(['REGN.csv'], FLAGS.batch_size, FLAGS.sequence_length, train_pct=1.0)
    helper.rewind()
    totals = [0,0,0,0]
    while True:
        X, y = helper.next_batch('train', verbose=True)
        if (X is None) or (y is None):
            break
        predictions, s = do_inference(FLAGS.server, FLAGS.work_dir,
            FLAGS.concurrency, X, FLAGS.batch_size, y)
        print('tp=%d, tn=%d, fp=%d, fn=%d' % (s[0], s[1], s[2], s[3]))
        for i in range(4):
            totals[i] += s[i]
    s = totals
    print('Total scores: tp=%d, tn=%d, fp=%d, fn=%d' % (s[0], s[1], s[2], s[3]))
    print(gridlstm.performance_metrics(*s))

def do_backtest():
    # Load csv files into BatchDataHelper
    csvfiles = glob.glob('csv/*.csv')
    helper = BatchDataHelper(sorted(csvfiles), FLAGS.batch_size, FLAGS.sequence_length,
        train_pct=1.0, max_csvfiles=10000)

    bar = progressbar.ProgressBar(max_value=helper.num_batches)
    print(helper.num_batches)

    # Run inference on all csvs
    totals = [0,0,0,0]
    batchcount = 0
    while True:
        X, y = helper.next_batch('train', verbose=True)
        if (X is None) or (y is None):
            break
        batchcount += 1
        bar.update(batchcount)
        predictions, s = do_inference(FLAGS.server, FLAGS.work_dir,
            FLAGS.concurrency, X, FLAGS.batch_size, y)

        for i in range(4):
            totals[i] += s[i]
    s = totals
    print('Total scores: tp=%d, tn=%d, fp=%d, fn=%d' % (s[0], s[1], s[2], s[3]))
    print(gridlstm.performance_metrics(*s))


def main(_):

    fh = logging.FileHandler('helitrax_client.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)
    gridlstm.logger = logger

    if not FLAGS.server:
        print('please specify server host:port')
        return
    command = FLAGS.command
    if command not in ['inference', 'backtest', 'test']:
        print('please specify valid command')
        return

    if command == 'test':
        do_test()

    elif command == 'inference':
        if not FLAGS.inputfile:
            print('please specify input json file')
            return

        # Load metadata file
        with open(FLAGS.inputfile) as f:
            metadata = json.load(f)

        if not metadata.has_key('numpy_datafile'):
            print('Error! missing key "numpy_datafile"')
            return 1
        X = np.load(str(metadata['numpy_datafile']))
        tickers = [str(t) for t in metadata['tickers']]
        predictions, _ = do_inference(FLAGS.server, FLAGS.work_dir,
            FLAGS.concurrency, X, FLAGS.batch_size) # Note - not providing y,
                                                    # so drop scores in return
        #print('shape predictions: %s' % predictions.shape)
        #print('len(tickers)=%d' % len(tickers))
        output = {}
        output['startdate'] = metadata['timestamps'][-1]
        output['predictions'] = {}
        for i in range(len(tickers)):
            print('%5s: %d' % (tickers[i], predictions[i]))
            output['predictions'][tickers[i]] = predictions[i]
        with open('predictions.json', 'w') as outfile:
            json.dump(output, outfile)

    elif command =='backtest':
        # Expects preprocessed csvs in folder "csv"
        do_backtest()

    else:
        print('Unexpected command: %s' % command)


if __name__ == '__main__':
  tf.app.run()
