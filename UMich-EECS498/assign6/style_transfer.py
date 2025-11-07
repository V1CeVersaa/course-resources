"""
Implements a style transfer in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""

import torch
from torch.nn import functional as F

# import torch.nn as nn
# from a6_helper import *


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Hello from style_transfer.py!")


def content_loss(content_weight, content_current, content_original):
    """
    Compute the content loss for style transfer.

    Inputs:
    - content_weight: Scalar giving the weighting for the content loss.
    - content_current: features of the current image; this is a PyTorch Tensor of shape
      (1, C_l, H_l, W_l).
    - content_original: features of the content image, Tensor with shape (1, C_l, H_l, W_l).

    Returns:
    - scalar content loss
    """
    ############################################################################
    # TODO: Compute the content loss for style transfer.                       #
    ############################################################################
    diff = content_current - content_original
    return content_weight * torch.sum(diff**2)
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################


def gram_matrix(features, normalize=True):
    """
    Compute the Gram matrix from features.

    Inputs:
    - features: PyTorch Tensor of shape (N, C, H, W) giving features for
      a batch of N images.
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)

    Returns:
    - gram: PyTorch Tensor of shape (N, C, C) giving the
      (optionally normalized) Gram matrices for the N input images.
    """
    ############################################################################
    # TODO: Compute the Gram matrix from features.                             #
    # Don't forget to implement for both normalized and non-normalized version #
    ############################################################################
    N, C, H, W = features.shape
    feats = features.view(N, C, H * W)
    gram = torch.bmm(feats, feats.transpose(1, 2))
    if normalize:
        gram = gram / (C * H * W)
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
    return gram


def style_loss(feats, style_layers, style_targets, style_weights):
    """
    Computes the style loss at a set of layers.

    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a PyTorch Tensor giving the Gram matrix of the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].

    Returns:
    - style_loss: A PyTorch Tensor holding a scalar giving the style loss.
    """
    ############################################################################
    # TODO: Computes the style loss at a set of layers.                        #
    # Hint: you can do this with one for loop over the style layers, and       #
    # should not be very much code (~5 lines).                                 #
    # You will need to use your gram_matrix function.                          #
    ############################################################################
    loss = torch.tensor(0.0, dtype=feats[0].dtype, device=feats[0].device)
    for weight, idx, target in zip(style_weights, style_layers, style_targets):
        current = gram_matrix(feats[idx].clone())
        loss += weight * F.mse_loss(current, target, reduction="sum")
    return loss
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################


def tv_loss(img, tv_weight):
    """
    Compute total variation loss.

    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.

    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    ############################################################################
    # TODO: Compute total variation loss.                                      #
    # Your implementation should be vectorized and not require any loops!      #
    ############################################################################
    x_diff = img[:, :, 1:, :] - img[:, :, :-1, :]
    y_diff = img[:, :, :, 1:] - img[:, :, :, :-1]
    return tv_weight * (torch.sum(x_diff**2) + torch.sum(y_diff**2))
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################


def guided_gram_matrix(features, masks, normalize=True):
    """
    Inputs:
      - features: PyTorch Tensor of shape (N, R, C, H, W) giving features for
        a batch of N images.
      - masks: PyTorch Tensor of shape (N, R, H, W)
      - normalize: optional, whether to normalize the Gram matrix
          If True, divide the Gram matrix by the number of neurons (H * W * C)

      Returns:
      - gram: PyTorch Tensor of shape (N, R, C, C) giving the
        (optionally normalized) guided Gram matrices for the N input images.
    """
    ##############################################################################
    # TODO: Compute the guided Gram matrix from features.                        #
    # Apply the regional guidance mask to its corresponding feature and          #
    # calculate the Gram Matrix. You are allowed to use one for-loop in          #
    # this problem.                                                              #
    ##############################################################################
    N, R, C, H, W = features.shape
    masked_features = features * masks.unsqueeze(2)
    masked_features = masked_features.view(N, R, C, H * W)
    gram = torch.matmul(masked_features, masked_features.transpose(-1, -2))
    if normalize:
        gram = gram / (C * H * W)
    return gram
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################


def guided_style_loss(feats, style_layers, style_targets, style_weights, content_masks):
    """
    Computes the style loss at a set of layers.

    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a PyTorch Tensor giving the guided Gram matrix of the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].
    - content_masks: List of the same length as feats, giving a binary mask to the
      features of each layer.

    Returns:
    - style_loss: A PyTorch Tensor holding a scalar giving the style loss.
    """
    ############################################################################
    # TODO: Computes the guided style loss at a set of layers.                 #
    ############################################################################
    loss = torch.tensor(0.0, dtype=feats[0].dtype, device=feats[0].device)
    for weight, idx, target in zip(style_weights, style_layers, style_targets):
        current = guided_gram_matrix(feats[idx], content_masks[idx])
        layer_loss = torch.nn.functional.mse_loss(current, target, reduction="sum")
        loss += weight * layer_loss
    return loss
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
