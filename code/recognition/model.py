import tensorflow as tf
import numpy as np


def create_fc_layer(inputs,neurons,mode):
    fc_layer = tf.contrib.layers.fully_connected(
        inputs=inputs,
        num_outputs=neurons
    )
    
    batch_norm = tf.contrib.layers.batch_norm(
        inputs=fc_layer)
    
    dropout = tf.layers.dropout(
        inputs=batch_norm,
        #rate=0.4,
        training=mode == tf.estimator.ModeKeys.TRAIN)
    
    relu_act = tf.nn.relu(
        features=dropout)

    return relu_act

def create_conv_layer(inputs,filters,kernels,padding,strides,pools,name,mode):
    #print(inputs,filters,kernels,padding,strides,pools,name,mode)
    conv_layer = tf.contrib.layers.conv2d(
        inputs=inputs,
        num_outputs=filters,
        kernel_size=kernels,
        padding=padding,
        stride=1,
        activation_fn=tf.nn.relu,
        normalizer_fn=tf.contrib.layers.batch_norm
    )
    
    #batch_norm = tf.contrib.layers.batch_norm(
    #    inputs=conv_layer)
    
    relu_act = tf.nn.relu(
        features=conv_layer)
    
    dropout = tf.layers.dropout(
        inputs=relu_act,
        #rate=0.4,
        training = mode == tf.estimator.ModeKeys.TRAIN)
    
    pooling = tf.layers.max_pooling2d(
        inputs=dropout,
        pool_size=pools,
        strides=strides)
    
    return pooling

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    n_conv = [(2,3,64),(2,5,128),(2,3,512)]
    n_fc = [256,512]
    
    # Input Layer
    input_layer = tf.reshape(tf.cast(features,tf.float32), [-1, 48, 48, 1])
    print(labels)
    #one_hot_labels = tf.one_hot(labels,7)
    
    conv_layer = input_layer
    
    for s,k,f in n_conv:
        conv_layer = create_conv_layer(
            inputs=conv_layer,
            filters=f,
            kernels=[k,k],
            padding="same",
            strides=s,
            pools=[2,2],
            name="conv"+str(f),
            mode=mode
            )
        
    
    
    fc_layer = tf.reshape(conv_layer,[-1,6*6*512])
    
    for n in n_fc:
        fc_layer = create_fc_layer(
            inputs=fc_layer,
            neurons=n,
            mode=mode
        )
        print(fc_layer)
        
    logits = tf.layers.dense(inputs=fc_layer, units=7)
    
    softmax = tf.nn.softmax(logits, name="softmax_tensor")

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": softmax
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.005)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def eval_input_fn(features, labels=None, batch_size=None):
    """An input function for evaluation or prediction"""

    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert inputs to a tf.dataset object.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()


class Predictor:

    def __init__(self):
        self.clf = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="tmp/mnist_convnet_model")
        
    def predict(self, im):

        pred = self.clf.predict(input_fn=eval_input_fn(features=[im],
                         batch_size=1))

        return pred

    

#
emotion_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="tmp/mnist_convnet_model")


