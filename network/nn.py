import tensorflow as tf
import numpy as np

def leaky_relu(x, alpha=0.01):
    return tf.maximum(alpha * x, x)

def retrive_var(scopes):
    var = []
    for scope in scopes:
        var += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=scope)
    return var

def gumbel_softmax(logits, gamma, eps=1e-20):
    U = tf.random_uniform(tf.shape(logits))
    G = -tf.log(-tf.log(U + eps) + eps)
    return tf.nn.softmax((logits + G) / gamma)

def softsample_word(dropout, proj_W, proj_b, embedding, gamma):

    def loop_func(output):
        output = tf.nn.dropout(output, dropout)
        logits = tf.matmul(output, proj_W) + proj_b
        prob = gumbel_softmax(logits, gamma)
        inp = tf.matmul(prob, embedding)
        return inp, prob

    return loop_func

def argmax_word(dropout, proj_W, proj_b, embedding):

    def loop_func(output):
        output = tf.nn.dropout(output, dropout)
        logits = tf.matmul(output, proj_W) + proj_b
        word = tf.argmax(logits, axis=1)
        inp = tf.nn.embedding_lookup(embedding, word)
        return inp, word

    return loop_func
