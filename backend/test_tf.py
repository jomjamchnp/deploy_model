import tensorflow as tf
import os
tf.compat.v1.disable_eager_execution()
def no_gpu():
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    # Creates a graph.
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)
    # Creates a session with log_device_placement set to True.
    config = tf.compat.v1.ConfigProto(
        device_count = {'GPU': 0}
    )
    sess = tf.compat.v1.Session(config=config)
    #config gpu list
    # Runs the op.
    print(sess.run(c))

no_gpu()