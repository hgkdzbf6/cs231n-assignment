from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        # X: N, D1
        # W1: D1, D2
        # b1: D2
        # W2: D2, D3
        # b2: D3
        self.params['W1'] = np.random.randn(input_dim, hidden_dim)*weight_scale
        self.params['b1'] = np.zeros(hidden_dim) # np.random.randn(hidden_dim)*weight_scale
        self.params['W2'] = np.random.randn(hidden_dim,num_classes)*weight_scale
        self.params['b2'] = np.zeros(num_classes) # np.random.randn(num_classes)*weight_scale
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        h = X.dot(self.params['W1']) + self.params['b1']
        h_ReLU = np.maximum(0,h)
        scores = h_ReLU.dot(self.params['W2'])+self.params['b2']
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        scores = scores - np.max(scores, axis=1, keepdims=True)
        Z = np.sum(np.exp(scores),axis=1,keepdims=True)
        log_probs = scores - np.log(Z)
        N = X.shape[0]
        probs = np.exp(log_probs)
        loss = -np.sum(log_probs[np.arange(N),y])/N + 0.5*self.reg* np.sum(self.params['W2']**2) +0.5* self.reg*np.sum(self.params['W1']**2)
        ds = probs.copy()
        ds[np.arange(N), y] -= 1
        ds /= N
        # ds += 0.5 * self.params['W2'] + 0.5 * self.params['W1']
        grads['W2'] = h_ReLU.T.dot(ds).reshape(self.params['W2'].shape) + self.reg * self.params['W2']
        grads['b2'] = np.sum(ds,axis=0).reshape(self.params['b2'].shape)
        dh_Relu = ds.dot(self.params['W2'].T).reshape(h_ReLU.shape)
        dh = np.where(h>0, dh_Relu, 0)
        grads['b1'] = np.sum(dh,axis=0).reshape(self.params['b1'].shape)
        # print(X.shape)
        # X_temp = np.reshape(X,N,)
        grads['W1'] = X.T.dot(dh).reshape(self.params['W1'].shape) + self.reg*self.params['W1']
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        # input_dim: D
        # Hidden1: H1,H2
        # class: C
        last_item = input_dim
        for i,item in enumerate (hidden_dims):
          key_W = 'W' + str(i+1)
          key_b = 'b' + str(i+1)
          key_gamma = 'gamma'+ str(i+1)
          key_beta = 'beta'+str(i+1)
          self.params[key_W] = np.random.randn(last_item, item) * weight_scale
          self.params[key_b] = np.zeros(item)
          if self.use_batchnorm:
            self.params[key_gamma] = np.ones(item)
            self.params[key_beta] = np.zeros(item)
          last_item = item
        key_W = 'W' + str(self.num_layers)
        key_b = 'b' + str(self.num_layers)
        self.params[key_W] = np.random.randn(last_item, num_classes) * weight_scale
        self.params[key_b] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            print(k,v.shape)
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        input_layer = X
        layer_cache = {}
        drop_cache = {}
        # 0 1
        for i in range(self.num_layers-1):
          Wi = 'W' + str(i+1)
          bi = 'b' + str(i+1)
          layer_index = i+1
          # 然后batchnorm
          if self.use_batchnorm:
            gammai = 'gamma'+ str(i+1)
            betai = 'beta'+str(i+1)
            input_layer, layer_cache[layer_index] = affine_bn_relu_forward(input_layer,self.params[Wi],self.params[bi],self.params[gammai],self.params[betai],self.bn_params[i])
          else:
            input_layer, layer_cache[layer_index] = affine_relu_forward(input_layer,self.params[Wi],self.params[bi])
          # 最后激活函数
          if self.use_dropout:
            input_layer, drop_cache[layer_index] = dropout_forward(input_layer,self.dropout_param)
        # 3 
        Wi = 'W' + str(self.num_layers)
        bi = 'b' + str(self.num_layers)
        output_layer, layer_cache[self.num_layers] = affine_forward(input_layer, self.params[Wi], self.params[bi])
        scores = output_layer
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # print(len(layer_cache[0][0][0]))

        loss, dscores = softmax_loss(scores, y)
        dhout = dscores
        Wi = 'W' + str(self.num_layers)
        bi = 'b' + str(self.num_layers)#3
        loss = loss + 0.5 * self.reg * np.sum(self.params[Wi]**2)
        dx, dw, db = affine_backward(dhout, layer_cache[self.num_layers])
        grads[Wi] = dw + self.reg*self.params[Wi]
        grads[bi] = db
        dhout = dx
        for i in range(self.num_layers-1):
          layer_index = self.num_layers - i - 1
          Wi = 'W' + str(layer_index)
          bi = 'b' + str(layer_index)
          loss = loss + 0.5 * self.reg * np.sum(self.params[Wi]**2)
          if self.dropout_param:
            dhout = dropout_backward(dhout, drop_cache[layer_index])

          if self.use_batchnorm:
            dx, dw, db, dgamma, dbeta = affine_bn_relu_backward(dhout,layer_cache[layer_index])
          else:
            dx, dw, db = affine_relu_backward(dhout, layer_cache[layer_index])
          grads[Wi] = dw + self.reg * self.params[Wi]
          grads[bi] = db
          if self.use_batchnorm:
            gammai = 'gamma' + str(layer_index)
            betai = 'beta' + str(layer_index)
            grads[gammai] = dgamma
            grads[betai] = dbeta
          dhout = dx
        
        # Cast all parameters to the correct datatype
        # for k, v in grads.items():
        #     print('grad',k,v.shape)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
