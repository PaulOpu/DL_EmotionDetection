import tensorflow as tf
import numpy as np

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



emotion_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="tmp/mnist_convnet_model")