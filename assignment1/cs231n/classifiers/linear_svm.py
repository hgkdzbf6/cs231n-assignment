import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]   # 10
  num_train = X.shape[0]     # 500
  loss = 0.0
  # 3073*10
  # print(dW.shape, X.shape, W.shape)
  for i in xrange(num_train):   # 500
    scores = X[i].dot(W)        # 10
    correct_class_score = scores[y[i]]
    # margin = scores - correct_class_score + 1
    # 问题： 这样的话，怎么去掉相同的元素呢？
    # margin[y[i]] = 0
    # margin = np.maximum(0,margin)
    # 问题： 这样的话，怎么更新dW呢？
    # dW += X[i].T.dot()
    for j in xrange(num_classes):    # 10
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i].T
        dW[:,y[i]] += -X[i].T

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW

def L_i_vectorized(x,y,W):
  delta = 1.0
  scores = W.dot(x)
  margins = np.maximum(0,scores - scores[y]+delta)
  margins[y] = 0
  loss_i = np.sum(margins)
  return loss_i

def L(X,y,W):
  loss = 0.0
  num_train = X.shape[1]
  num_classes = W.shape[0]
  for i in range(num_classes):
    loss+= L_i_vectorized(X[i],y,W)
  loss /= num_train
  loss += np.sum(W*W)
  return loss

def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  num_train = X.shape[0]
  num_classes = W.shape[1]
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  S = X.dot(W)    # 500*10
  # print(S.shape)
  # 这里参看numpy的用法，应该很简单的把qwq
  S_y = S[range(num_train), list(y)].reshape(-1,1)    # 500
  # 这边是广播了
  mid = np.maximum(S-S_y+1,0)                         # 500*10
  mid[range(num_train), list(y)]=0
  L_i = np.sum(mid,axis=1)  # 先列再行，得到的是列向量
  loss = np.average(L_i) + np.sum(np.square(W))
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  # 如何算j \ne y_i?
  coeff_mat = np.zeros((num_train, num_classes))
  # 小于0的不要，大于0的先加上一个1
  coeff_mat[mid > 0] = 1
  # 真值不参与计算
  coeff_mat[range(num_train), list(y)] = 0
  # 对每个真值都减去一行的所有非零值
  coeff_mat[range(num_train), list(y)] = -np.sum(coeff_mat, axis=1)
  dW += X.T.dot(coeff_mat)
  dW /=num_train
  dW += 2 * reg * W
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
