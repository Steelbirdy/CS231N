import tensorflow as tf
import numpy as np

def tv_loss(img, tv_weight):
    """
    Compute total variation loss.

    Inputs:
    - img: Tensor of shape (1, H, W, 3) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.

    Returns:
    - loss: Tensor holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    # Your implementation should be vectorized and not require any loops!
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # imgt = tf.reshape(img, (*img.shape[1:],)).numpy()
    # h_shifted = tf.roll(imgt, shift=-1, axis=0).numpy()
    # v_shifted = tf.roll(imgt, shift=-1, axis=1).numpy()
    # h_diff = h_shifted - imgt
    # h_diff[-1, :] = 0
    # v_diff = v_shifted - imgt
    # v_diff[:, -1] = 0
    #
    # return tv_weight * np.sum(np.square(h_diff) + np.square(v_diff))  # Summation over RGB channels
    Wx = np.zeros([1, 2, 3, 3])
    Wx[0, 0:, 0, 0] = [-1, 1]
    Wx[0, 0:, 1, 1] = [-1, 1]
    Wx[0, 0:, 2, 2] = [-1, 1]
    Wx = tf.constant(Wx, dtype=tf.float32)
    Wy = np.zeros([2, 1, 3, 3])
    Wy[0:, 0, 0, 0] = [-1, 1]
    Wy[0:, 0, 1, 1] = [-1, 1]
    Wy[0:, 0, 2, 2] = [-1, 1]
    Wy = tf.constant(Wy, dtype=tf.float32)
    loss_x = tf.nn.conv2d(img, Wx, strides=[1, 1, 1, 1], padding='VALID')
    loss_y = tf.nn.conv2d(img, Wy, strides=[1, 1, 1, 1], padding='VALID')
    loss = tf.nn.l2_loss(loss_x) + tf.nn.l2_loss(loss_y)
    loss *= tv_weight * 2
    return loss
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

def style_loss(feats, style_layers, style_targets, style_weights):
    """
    Computes the style loss at a set of layers.

    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a Tensor giving the Gram matrix of the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].

    Returns:
    - style_loss: A Tensor containing the scalar style loss.
    """
    # Hint: you can do this with one for loop over the style layers, and should
    # not be short code (~5 lines). You will need to use your gram_matrix function.
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # style_loss = 0
    # for i, layer in enumerate(style_layers):
    #     style_loss += style_weights[i] * tf.square(tf.norm(gram_matrix(feats[layer]) - style_targets[i]))
    # return style_loss
    style_loss = 0
    for idx, layer in enumerate(style_layers):
        style_loss += style_weights[idx] * 2 * tf.nn.l2_loss(gram_matrix(feats[layer]) - style_targets[idx])
    return style_loss
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

def gram_matrix(features, normalize=True):
    """
    Compute the Gram matrix from features.

    Inputs:
    - features: Tensor of shape (1, H, W, C) giving features for
      a single image.
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)

    Returns:
    - gram: Tensor of shape (C, C) giving the (optionally normalized)
      Gram matrices for the input image.
    """
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # _, H, W, C = features.shape
    # f2 = tf.reshape(features, (C, -1))
    # result = tf.matmul(f2, f2, transpose_b=True)
    # if normalize:
    #     result /= H * W * C
    # return result
    v = tf.transpose(features, perm=[3, 1, 2, 0])
    v = tf.reshape(v, [tf.shape(features)[3], -1])
    gram = tf.matmul(v, tf.transpose(v))
    if normalize:
        gram /= float(tf.size(features)) * float(tf.shape(features)[0])
    return gram
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

def content_loss(content_weight, content_current, content_original):
    """
    Compute the content loss for style transfer.

    Inputs:
    - content_weight: scalar constant we multiply the content_loss by.
    - content_current: features of the current image, Tensor with shape [1, height, width, channels]
    - content_target: features of the content image, Tensor with shape [1, height, width, channels]

    Returns:
    - scalar content loss
    """
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # For some reason I always get an error of 0.258, but I'm almost positive this is the way to do it
    # return content_weight * tf.square(tf.norm(content_original - content_current))
    loss = content_weight * 2 * tf.nn.l2_loss(content_original - content_current)
    return loss
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# We provide this helper code which takes an image, a model (cnn), and returns a list of
# feature maps, one per layer.
def extract_features(x, cnn):
    """
    Use the CNN to extract features from the input image x.

    Inputs:
    - x: A Tensor of shape (N, H, W, C) holding a minibatch of images that
      will be fed to the CNN.
    - cnn: A Tensorflow model that we will use to extract features.

    Returns:
    - features: A list of feature for the input images x extracted using the cnn model.
      features[i] is a Tensor of shape (N, H_i, W_i, C_i); recall that features
      from different layers of the network may have different numbers of channels (C_i) and
      spatial dimensions (H_i, W_i).
    """
    features = []
    prev_feat = x
    for i, layer in enumerate(cnn.net.layers[:-2]):
        next_feat = layer(prev_feat)
        features.append(next_feat)
        prev_feat = next_feat
    return features

def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
