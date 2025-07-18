import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

# Based on the implementation by an author of the ArcFace paper
# https://github.com/4uiiurz1/keras-arcface/blob/master/keras_arcface/layers.py
class ArcMarginProduct(Layer):
    """
    Implements the Additive Angular Margin Loss (ArcFace) as a Keras Layer.
    """
    def __init__(self, n_classes, s=30.0, m=0.50, regularizer=None, **kwargs):
        super(ArcMarginProduct, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s  # Scale factor
        self.m = m  # Margin
        self.regularizer = tf.keras.regularizers.get(regularizer)

    def build(self, input_shape):
        # The weight matrix (W) for the classes. This is the "class centers".
        # The shape is (embedding_size, number_of_classes)
        self.W = self.add_weight(
            name='W',
            shape=(input_shape[-1], self.n_classes),
            initializer='glorot_uniform',
            trainable=True,
            regularizer=self.regularizer
        )
        super(ArcMarginProduct, self).build(input_shape)

    def call(self, inputs):
        # The 'inputs' are the feature embeddings (x) from the backbone model.
        # The shape is (batch_size, embedding_size)
        x = inputs

        # 1. Normalize the feature embeddings (x) and the weights (W)
        x_norm = tf.nn.l2_normalize(x, axis=1)
        W_norm = tf.nn.l2_normalize(self.W, axis=0)

        # 2. Calculate the cosine similarity (cos(theta))
        # This is the dot product between normalized x and W.
        # Resulting shape is (batch_size, n_classes)
        cosine = tf.matmul(x_norm, W_norm)

        return cosine # We will handle the margin addition in the loss function

    def get_config(self):
        config = super(ArcMarginProduct, self).get_config()
        config.update({
            'n_classes': self.n_classes,
            's': self.s,
            'm': self.m,
            'regularizer': tf.keras.regularizers.serialize(self.regularizer),
        })
        return config

def aamsoftmax_loss(y_true, y_pred, s=30.0, m=0.50):
    """
    Additive Angular Margin Softmax Loss as a custom loss function.
    
    Parameters:
    - y_true: Tensor or NumPy array of shape (batch_size, n_classes), where each row is a one-hot encoded vector 
      representing the true class labels. Must be of type float32 or float64.
    - y_pred: Tensor or NumPy array of shape (batch_size, n_classes), containing the cosine similarities (logits) 
      from the ArcMarginProduct layer. Must be of type float32 or float64.
    - s: Float. Scale factor for the logits. Default is 30.0.
    - m: Float. Additive angular margin. Default is 0.50.
    
    Returns:
    - loss: Tensor containing the computed categorical cross-entropy loss.
    """
    # Clip the cosine values to prevent numerical instability with acos
    cosine = tf.clip_by_value(y_pred, -1.0 + K.epsilon(), 1.0 - K.epsilon())

    # Calculate the angle (theta)
    theta = tf.acos(cosine)

    # Add the margin (m) to the angle of the correct class
    # y_true is one-hot, so multiplying it by theta gives us the theta_yi
    # and adding m to it effectively does theta_yi + m
    target_logits = tf.cos(theta + m)

    # Create the final logits: use the modified logits for the correct class
    # and the original logits for the other classes.
    # y_true is one-hot, so (1.0 - y_true) is a mask for the incorrect classes.
    final_logits = y_pred * (1.0 - y_true) + target_logits * y_true

    # Scale the logits
    scaled_logits = final_logits * s

    # Compute the cross-entropy loss
    loss = tf.keras.losses.categorical_crossentropy(y_true, scaled_logits, from_logits=True)
    return loss