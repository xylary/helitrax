import sys
import threading
from grpc.beta import implementations
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow_serving.apis import classification_pb2
from tensorflow_serving.apis import classification_service_pb2
from tensorflow_serving.example import mnist_input_data
from gridlstm import BatchDataHelper

tf.app.flags.DEFINE_integer('concurrency', 1,
                            'maximum number of concurrent inference requests')
tf.app.flags.DEFINE_string('server', '', 'PredictionService host:port')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory. ')
FLAGS = tf.app.flags.FLAGS

def do_inference(hostport, work_dir, concurrency, num_tests):
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
    #test_data_set = mnist_input_data.read_data_sets(work_dir).test
    host, port = hostport.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.PredictionServiceStub(channel)
    #result_counter = _ResultCounter(num_tests, concurrency)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'helitrax' # Note this is associated with --model_name argument to tensorflow_model_server
    request.model_spec.signature_name = signature_constants.CLASSIFY_INPUTS

    helper = BatchDataHelper('REGN.csv', batch_size, sequence_length)
    X_batch, y_batch = helper.next_batch('train')

    request.inputs[signature_constants.CLASSIFY_INPUTS].CopyFrom(
        tf.contrib.util.make_tensor_proto(X_batch, shape=[20, 200, 79]))

    result = stub.Classify(request)
    print(result)
    
    return result



def main():
    if FLAGS.num_tests > 10000:
        print('num_tests should not be greater than 10k')
        return
    if not FLAGS.server:
        print('please specify server host:port')
        return

    do_inference(FLAGS.server, FLAGS.work_dir,
        FLAGS.concurrency)

if __name__ == '__main__':
  tf.app.run()
