import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  D,C = W.shape   # 3072,10
  N = X.shape[0]  # 500

  for i in range(N):
    scores = X[i].dot(W)
    scores = scores - np.max(scores)
    # print(scores)
    # print(scores.shape)
    correct_class_score = scores[y[i]]
    # dW += (X[i].T - X[y[i]].T)*np.exp(scores - correct_class_score)
    sum = 0
    for j in range(C):
      sum += np.exp(scores[j])
      softmax_output = np.exp(scores[j])/np.sum(np.exp(scores))
      if j == y[i]:
          dW[:,j] += (-1 + softmax_output) *X[i] 
      else: 
          dW[:,j] += softmax_output *X[i] 
    margin = - np.log(np.exp(correct_class_score)/sum)
    # margin = - np.log(np.exp(correct_class_score)/np.sum(np.exp(scores)))
    loss+=margin

  loss /=N
  dW /=N
  loss+=reg*np.sum(W**2)
  dW += 2*reg*W
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  # loss = L(X,y,W)
  return loss, dW

def L_i(x,y,W):
  scores_i = x.dot(W)
  correct_class_score_i = scores_i[y]
  D = W.shape[1]
  loss_i = 0.0
  sum = 0.0
  for j in range(D):
    sum += np.exp(scores_i[j])
  loss_i += -np.log(np.exp(correct_class_score_i)/sum)
  return loss_i

def L_i_vectorized(x,y,W):
  loss_i = 0.0
  scores = W.dot(x)
  correct_class_score = scores[y]
  sum = np.sum(np.exp(scores))
  loss_i = -np.log(np.exp(correct_class_score)/sum)
  return loss_i

def L(X, y, W):
  loss = 0.0
  num_train = X.shape[0]
  for i in range(num_train):
    loss+=L_i(X[i], y[i],W)
  loss /=num_train
  loss+=np.sum(W**2)
  return loss

def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  S = X.dot(W) # 500*10
  num_train, num_classes = S.shape
  S = S - np.max(S, axis= 1).reshape(-1,1)
  S_y = S[range(num_train), list(y)].reshape(-1,1)
  sum = np.sum(np.exp(S),axis=1).reshape(-1,1)
  exp_S_y = np.exp(S_y).reshape(-1,1)
  softmax_output = np.exp(S)/np.sum(np.exp(S),axis=1).reshape(-1,1)
  loss += np.sum(
    -np.log(exp_S_y/sum))
  # dW += X.T.dot((exp_S_y/sum).reshape(-1,1))
  test = softmax_output.copy()
  test[range(num_train), list(y)] += -1
  dW = (X.T).dot(test)
  dW /= num_train
  loss /= num_train
  loss += reg * np.sum(W * W)
  dW += 2*np.sum(W)
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

