"""Copyright 2019 Google LLC

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License.

"""
import tensorflow as tf
from neural_process.util import batch_mlp


def uniform_attention(q, v):
    """Uniform attention. Equivalent to np.

    Args:
        q: queries. tensor of shape [B,m,d_k].
        v: values. tensor of shape [B,n,d_v].
        
    Returns:
        tensor of shape [B,m,d_v].
    """
    total_points = tf.shape(q)[1]
    rep = tf.reduce_mean(v, axis=1, keepdims=True)    # [B,1,d_v]
    rep = tf.tile(rep, [1, total_points, 1])
    return rep

def laplace_attention(q, k, v, scale, normalise):
    """Computes laplace exponential attention.

    Args:
        q: queries. tensor of shape [B,m,d_k].
        k: keys. tensor of shape [B,n,d_k].
        v: values. tensor of shape [B,n,d_v].
        scale: float that scales the L1 distance.
        normalise: Boolean that determines whether weights sum to 1.
        
    Returns:
        tensor of shape [B,m,d_v].
    """
    k = tf.expand_dims(k, axis=1)    # [B,1,n,d_k]
    q = tf.expand_dims(q, axis=2)    # [B,m,1,d_k]
    unnorm_weights = - tf.abs((k - q) / scale)    # [B,m,n,d_k]
    unnorm_weights = tf.reduce_sum(unnorm_weights, axis=-1)    # [B,m,n]
    if normalise:
        weight_fn = tf.nn.softmax
    else:
        weight_fn = lambda x: 1 + tf.tanh(x)
    weights = weight_fn(unnorm_weights)    # [B,m,n]
    rep = tf.einsum('bik,bkj->bij', weights, v)    # [B,m,d_v]
    return rep


def dot_product_attention(q, k, v, normalise):
    """Computes dot product attention.

    Args:
        q: queries. tensor of    shape [B,m,d_k].
        k: keys. tensor of shape [B,n,d_k].
        v: values. tensor of shape [B,n,d_v].
        normalise: Boolean that determines whether weights sum to 1.
        
    Returns:
        tensor of shape [B,m,d_v].
    """
    d_k = tf.shape(q)[-1]
    scale = tf.sqrt(tf.cast(d_k, tf.float32))
    unnorm_weights = tf.einsum('bjk,bik->bij', k, q) / scale    # [B,m,n]
    if normalise:
        weight_fn = tf.nn.softmax
    else:
        weight_fn = tf.sigmoid
    weights = weight_fn(unnorm_weights)    # [B,m,n]
    rep = tf.einsum('bik,bkj->bij', weights, v)    # [B,m,d_v]
    return rep


def multihead_attention(q, k, v, num_heads=8):
    """Computes multi-head attention.

    Args:
        q: queries. tensor of    shape [B,m,d_k].
        k: keys. tensor of shape [B,n,d_k].
        v: values. tensor of shape [B,n,d_v].
        num_heads: number of heads. Should divide d_v.
        
    Returns:
        tensor of shape [B,m,d_v].
    """
    d_k = q.get_shape().as_list()[-1]
    d_v = v.get_shape().as_list()[-1]
    head_size = d_v / num_heads
    key_initializer = tf.random_normal_initializer(stddev=d_k**-0.5)
    value_initializer = tf.random_normal_initializer(stddev=d_v**-0.5)
    rep = tf.constant(0.0)
    for h in range(num_heads):
        o = dot_product_attention(
                tf.layers.Conv1D(head_size, 1, kernel_initializer=key_initializer,
                                     name='wq%d' % h, use_bias=False, padding='VALID')(q),
                tf.layers.Conv1D(head_size, 1, kernel_initializer=key_initializer,
                                     name='wk%d' % h, use_bias=False, padding='VALID')(k),
                tf.layers.Conv1D(head_size, 1, kernel_initializer=key_initializer,
                                     name='wv%d' % h, use_bias=False, padding='VALID')(v),
                normalise=True)
        rep += tf.layers.Conv1D(d_v, 1, kernel_initializer=value_initializer,
                                            name='wo%d' % h, use_bias=False, padding='VALID')(o)
    return rep

class Attention(object):
    """The Attention module."""

    def __init__(self, rep, output_sizes, att_type, scale=1., normalise=True,
                             num_heads=8):
        """Create attention module.

        Takes in context inputs, target inputs and
        representations of each context input/output pair
        to output an aggregated representation of the context data.
        Args:
            rep: transformation to apply to contexts before computing attention. 
                    One of: ['identity','mlp'].
            output_sizes: list of number of hidden units per layer of mlp.
                    Used only if rep == 'mlp'.
            att_type: type of attention. One of the following:
                    ['uniform','laplace','dot_product','multihead']
            scale: scale of attention.
            normalise: Boolean determining whether to:
                    1. apply softmax to weights so that they sum to 1 across context pts or
                    2. apply custom transformation to have weights in [0,1].
            num_heads: number of heads for multihead.
        """
        self._rep = rep
        self._output_sizes = output_sizes
        self._type = att_type
        self._scale = scale
        self._normalise = normalise
        if self._type == 'multihead':
            self._num_heads = num_heads

    def __call__(self, x1, x2, r):
        """Apply attention to create aggregated representation of r.

        Args:
            x1: tensor of shape [B,n1,d_x].
            x2: tensor of shape [B,n2,d_x].
            r: tensor of shape [B,n1,d].
            
        Returns:
            tensor of shape [B,n2,d]

        Raises:
            NameError: The argument for rep/type was invalid.
        """
        if self._rep == 'identity':
            k, q = (x1, x2)
        elif self._rep == 'mlp':
            # Pass through MLP
            k = batch_mlp(x1, self._output_sizes, "attention")
            q = batch_mlp(x2, self._output_sizes, "attention")
        else:
            raise NameError("'rep' not among ['identity','mlp']")

        if self._type == 'uniform':
            rep = uniform_attention(q, r)
        elif self._type == 'laplace':
            rep = laplace_attention(q, k, r, self._scale, self._normalise)
        elif self._type == 'dot_product':
            rep = dot_product_attention(q, k, r, self._normalise)
        elif self._type == 'multihead':
            rep = multihead_attention(q, k, r, self._num_heads)
        else:
            raise NameError(("'att_type' not among ['uniform','laplace','dot_product'"
                                             ",'multihead']"))

        return rep