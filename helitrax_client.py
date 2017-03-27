import sys
import threading
from grpc.beta import implementations
import numpy as np
import tensorflow as tf
sys.path.append('/serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server_test_client.runfiles/tf_serving/')
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow.python.framework import dtypes
from gridlstm import BatchDataHelper

tf.app.flags.DEFINE_integer('concurrency', 1,
                            'maximum number of concurrent inference requests')
tf.app.flags.DEFINE_integer('batch_size', 200,
                            'batch size model expects')
tf.app.flags.DEFINE_integer('sequence_length', 20,
                            'sequence length model expects')
tf.app.flags.DEFINE_string('server', '', 'PredictionService host:port')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory. ')
tf.app.flags.DEFINE_string('inputfile', '', 'Data intput filename')
tf.app.flags.DEFINE_string('tickerfile', '', 'Ordered list of tickers matching input')
tf.app.flags.DEFINE_bool('test', False, 'test mode')
FLAGS = tf.app.flags.FLAGS

def do_inference(hostport, work_dir, concurrency, X, batch_size=200):
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


    num_batches = X.shape[1] / batch_size
    X_batches = np.hsplit(X, num_batches)
    results = []
    for X in X_batches:
        request.inputs[signature_constants.CLASSIFY_INPUTS].CopyFrom(
            tf.contrib.util.make_tensor_proto(X, dtype=dtypes.float32, shape=[20, 200, 79]))
        result = stub.Predict(request, 100.0)
        print(result)
        results.append(result)

    return np.stack(results)



def main(_):
    if not FLAGS.server:
        print('please specify server host:port')
        return
    if FLAGS.test:
        helper = BatchDataHelper(['REGN.csv'], FLAGS.batch_size, FLAGS.sequence_length)
        X, y = helper.next_batch('train')
    else:
        if not FLAGS.inputfile:
            print('please specify input data file')
            return
        if not FLAGS.tickerfile:
            print('please specify ticker file')
            return

        X = np.load(FLAGS.inputfile)

    results = do_inference(FLAGS.server, FLAGS.work_dir,
        FLAGS.concurrency, X, FLAGS.batch_size)

if __name__ == '__main__':
  tf.app.run()
