from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    #Reshape input to [batch size, width, height, channels]
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    #Create Convolutional layer 1
    conv1 = tf.layers.conv2d(
        inputs = input_layer,
        filters = 32,
        kernel_size = [5, 5],
        padding = "same", #Output tensor has same h & w as input
        activation = tf.nn.relu) #Introduce non-linearity

    #Create Pooling layer 1, reduces size of images by 50%
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2, 2], #Filter size
        strides=2)

    #Create Convolutional layer 2
    conv2 = tf.layers.conv2d(
        inputs = pool1,
        filters = 64, #Twice as many filters
        kernel_size = [5,5],
        padding = "same",
        activation = tf.nn.relu)

    #Create Pooling layer 2
    pool2 = tf.layers.max_pooling2d(
        inputs = conv2,
        pool_size=[2, 2],
        strides=2)

    #Flatten feature map (Pooling Layer 2), w/h are now 25% of original
    #due to pooling layers, channels are now 64 due to convolutional layer filters.
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    #Create first dense layer
    dense = tf.layers.dense(
        inputs = pool2_flat,
        units = 1024, #Number of neurons
        activation = tf.nn.relu)

    #Randomly cut 40% of neurons per training pass to curb overfitting (dependency)
    dropout = tf.layers.dropout(
        inputs = dense,
        rate = 0.4,
        training = mode == tf.estimator.ModeKeys.TRAIN)

    #Create final dense layer (one neuron per possible output variable)
    logits = tf.layers.dense(inputs = dropout, units = 10)

    #Convert final tensor to dict with classes and probabilities,
    #current tensor shape is [batch_size, 10]
    predictions = {
        "classes" : tf.argmax(input = logits, axis = 1), #find class in logits[1]
        "probabilities" : tf.nn.softmax(logits, name = "softmax_tensor") #find probabilities and name tensor
    }

    #Return EstimatorSpec object with dict if in predict mode.
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    #Calculator loss (cross entropy model)
    onehot_labels = tf.one_hot(indices = tf.cast(labels, tf.int32), depth = 10)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels = onehot_labels, logits = logits)

    #Configure the training operation
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step = tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op)

    #Configure the evaluation operation
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            "accuracy" : tf.metrics.accuracy(
                labels = labels, predictions = predictions["classes"])}
        return tf.estimator.EstimatorSpec(
            mode = mode, loss = loss, eval_metric_ops = eval_metric_ops)


def main(unused_argv):

    #Load training and testing data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images #imports as numpy array
    train_labels = np.asarray(mnist.train.labels, dtype = np.int32)
    eval_data = mnist.test.images # imports as numpy array
    eval_labels = np.asarray(mnist.test.labels, dtype = np.int32)

    #Create Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn = cnn_model_fn, model_dir = "/tmp/mnist_convnet_model")
        #Point model function to defined cnn function
        #Set directory for checkpointing

    #Set up Logging Hook
    tensors_to_log = {"probabilities" : "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter = 50)
        #Log tensors every 50 steps of training

    #Train the Model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {'x' : train_data}, #Pass images to x as dict
        y = train_labels, #Pass labels to y
        batch_size = 100, #Train in batches of 100 per step
        num_epochs = None, #Train until specified number of steps is reached
        shuffle = True) #Shuffle training data

    #Pass data to estimator object
    mnist_classifier.train(
        input_fn = train_input_fn,
        steps = 20000,
        hooks = [logging_hook])

    #Create evaluation function
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {'x' : eval_data}, #Pass test data to x
        y = eval_labels, #Pass test labels to y
        num_epochs = 1, #Evaluate one epoch then return result
        shuffle = False) #Iterate through data sequentially

    #Run evauluation function and pass to variable
    eval_results = mnist_classifier.evaluate(input_fn = eval_input_fn)

    #Print results
    print(eval_results)

if __name__ == "__main__":
    tf.app.run()






