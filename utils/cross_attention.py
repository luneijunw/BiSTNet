import tensorflow as tf
# from lingvo import compat as tf
from lingvo.core import py_utils
from lingvo.tasks.car import car_lib
from typing import Any,Optional,List,Union
import numpy as np

def GetShape(
    tensor: Any,  # anything that can be converted to a tf.Tensor
    ndims: Optional[int] = None
) -> Union[List[Union[int, tf.Tensor]], tf.Tensor]:
  """Returns tensor's shape as a list which can be unpacked, unlike tf.shape.
  If the tensor is unranked, and ndims is None, returns the shape as a Tensor.
  Otherwise, returns a list of values. Each element in the list is an int (when
  the corresponding dimension is static), or a scalar tf.Tensor (when the
  corresponding dimension is dynamic).
  Args:
    tensor: The input tensor.
    ndims: If not None, returns the shapes for the first `ndims` dimensions.
  """
  tensor = tf.convert_to_tensor(tensor)
  dynamic_shape = tf.shape(tensor)

  # Early exit for unranked tensor.
  if tensor.shape.ndims is None:
    if ndims is None:
      return dynamic_shape
    else:
      return [dynamic_shape[x] for x in range(ndims)]

  # Ranked tensor.
  if ndims is None:
    ndims = tensor.shape.ndims
  else:
    ndims = min(ndims, tensor.shape.ndims)

  # Return mixture of static and dynamic dims.
  static_shape = tensor.shape.as_list()
  shapes = [
      static_shape[x] if static_shape[x] is not None else dynamic_shape[x]
      for x in range(ndims)
  ]
  return shapes

class DeepFusionAligner(SinglePointAligner):
  """Align image features to point cloud features.
  For each point cloud feature (i.e., each pillar), we extract Multiple
  corresponding image features, and then weighted sum these image features.
  The weights come from a cross-attention module.
  """

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    for param in [
        p.q_embedding, p.k_embedding, p.v_embedding, p.attn_dropout,
        p.learnable_align_fc
    ]:
      assert param
      self.CreateChild(param.name, param)

  @classmethod
  def Params(cls, *arg, **kwargs):
    p = super().Params(*arg, **kwargs)
    # See DeepFusion (https://arxiv.org/pdf/2203.08195.pdf) Section 3.3
    # paragraph LearnableAlign and Figure 1 for details.
    p.Define('q_embedding', None, 'The embedding function for Q^l.')
    p.Define('k_embedding', None, 'The embedding function for K^c.')
    p.Define('v_embedding', None, 'The embedding function for V^c.')
    p.Define('learnable_align_fc', None,
             'The fully connected layer in the LearnableAlign module')
    p.Define(
        'attn_dropout', None, 'Attention Dropout Module. See'
        'DeepFusion (https://arxiv.org/pdf/2203.08195.pdf)'
        'Section 4.1 paragraph LearnableAlign for details.')
    return p

  def FProp(self, theta, image_features, feat_ratio, points_projected,
            dynamic_voxels, featurized_cell):
    """Align image features to point cloud features.
    Args:
      theta: A `.NestedMap` object containing variable values of this task.
      image_features: A float tensor with shape [batch_size, num_cameras, H, W,
        C] containing the features extracted from backbone network.
      feat_ratio: A float for indicating the ratio between the feature map size
        and original image size, assuming that assuming the height and width are
        scaled with the same ratio.
      points_projected: NestedMap of cameras_idx, points_in_best_camera, and
        mask.
      dynamic_voxels: A NestedMap corresponding to the output of running
        DynamicVoxelization on points_xyz.
      featurized_cell: A float tensor with shape [batch_size, pseudo_image_H,
        pseudo_image_W, C] containing the lidar feature extracted from backbone
        lidar feature extractor.
    Returns:
      image_features_cell: The image feature aligned with the point cloud
      feature.
    """
    image_features_cell = super().FProp(theta, image_features, feat_ratio,
                                        points_projected)
    # Compute (single-head) cross attention weights.
    # To illustrate the following tensor shape, we define B as batch size, N as
    # the maximum number of 3D points, and C (or C') as the number of channels.
    # featurized_cell_shape = py_utils.GetShape(featurized_cell)
    featurized_cell_shape = GetShape(featurized_cell)

    flatten_featurized_cell = tf.reshape(featurized_cell, [
        featurized_cell_shape[0], featurized_cell_shape[1] *
        featurized_cell_shape[2], featurized_cell_shape[3]
    ])  # with shape [B, pseudo_image_H * pseudo_image_W, C]
    featurized_cell4attention = tf.gather(
        flatten_featurized_cell, dynamic_voxels.indices,
        batch_dims=1)  # with shape [B, N, C]

    q = self.q_embedding(featurized_cell4attention)  # with shape [B, N, C']
    k = self.k_embedding(image_features_cell)  # with shape [B, N, C']
    v = self.v_embedding(image_features_cell)  # with shape [B, N, C']

    affinity = tf.einsum('bnc,bnc->bn', q, k) / tf.sqrt(
        tf.cast(q.shape[-1], tf.float32))  # with shape [B, N]
    invalid_mask = (points_projected.mask * (1 - dynamic_voxels.padding)) < 0.5
    affinity = tf.where_v2(invalid_mask, -tf.ones_like(affinity) * np.inf,
                           affinity)

    # Do softmax on affinity to compute attention weights.
    max_affinity = car_lib.BatchedUnsortedSegmentMax(affinity,
                                                     dynamic_voxels.indices,
                                                     dynamic_voxels.num_voxels)
    max_affinity = tf.gather(max_affinity, dynamic_voxels.indices, batch_dims=1)
    e_affinity = tf.exp(affinity - max_affinity)
    e_affinity_sum = car_lib.BatchedUnsortedSegmentSum(
        e_affinity,
        dynamic_voxels.indices,
        dynamic_voxels.num_voxels,
        batched_padding=tf.cast(invalid_mask, tf.float32))
    e_affinity_sum = tf.gather(
        e_affinity_sum, dynamic_voxels.indices, batch_dims=1)
    weights = (e_affinity * (1. - tf.cast(invalid_mask, tf.float32))) / (
        e_affinity_sum + 1e-3 * tf.cast(invalid_mask, tf.float32))
    weights = self.attn_dropout(weights)

    retrieved_output_flatten = tf.einsum('bn,bnc->bnc', weights, v)
    retrieved_output_flatten = self.learnable_align_fc(retrieved_output_flatten)
    image_features_cell = car_lib.BatchedUnsortedSegmentSum(
        retrieved_output_flatten,
        dynamic_voxels.indices,
        dynamic_voxels.num_voxels,
        batched_padding=tf.cast(invalid_mask, tf.float32))
    return image_features_cell