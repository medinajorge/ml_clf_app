import tensorflow as tf
from typing import Optional

@tf.keras.utils.register_keras_serializable(package='CustomLosses')
class FocalLoss(tf.keras.losses.Loss):
    """Focal Loss, adapted from
    - https://github.com/AdeelH/pytorch-multi-class-focal-loss/blob/master/focal_loss.py
    - https://github.com/artemmavrin/focal-loss/blob/master/src/focal_loss/_categorical_focal_loss.py

    Necessary for loading Inception models.

    Args:
        alpha (Optional[tf.Tensor]): Class weighting factor. Defaults to None.
        gamma (float): Focusing parameter. Defaults to 0.
        reduction (str): Reduction method ('mean', 'sum', 'none'). Defaults to 'mean'.
        from_logits (bool): Whether predictions are logits. Defaults to False.
        name (str): Loss name. Defaults to 'FocalLoss'.
    """

    def __init__(self,
                 alpha: Optional[tf.Tensor] = None,
                 gamma: float = 2.0,
                 reduction: str = 'mean',
                 from_logits: bool = False,
                 epsilon: float = 1e-7,
                 name: str = 'FocalLoss'):
        super().__init__(reduction=tf.keras.losses.Reduction.NONE, name=name)
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError("Reduction must be 'mean', 'sum', or 'none'.")
        self.alpha = tf.convert_to_tensor(alpha, dtype=tf.float32) if alpha is not None else None
        self.gamma = tf.convert_to_tensor(gamma, dtype=tf.float32)
        self.user_reduction = reduction
        self.from_logits = from_logits
        self.epsilon = epsilon

    # @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    #                               tf.TensorSpec(shape=(None, None), dtype=tf.float32)])
    @tf.function
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        y_true: tensor (int)
        y_pred: tensor (float)
        """
        if len(y_true.shape) > 1:
            y_true = y_true[:, 0]

        if y_true.dtype != tf.int32:
            y_true = tf.cast(y_true, tf.int32)

        if y_pred.shape.rank > 2:
            y_pred = tf.reshape(y_pred, [-1, y_pred.shape[-1]])

        if self.from_logits:
            logits = y_pred
            probs = tf.nn.softmax(y_pred, axis=-1)
            ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=logits)
        else:
            probs = y_pred
            log_softmax = tf.math.log(tf.clip_by_value(y_pred, self.epsilon, 1.0 - self.epsilon))
            log_pt = tf.gather(log_softmax, y_true, axis=1, batch_dims=1)
            ce = -log_pt

        probs = tf.gather(probs, y_true, axis=1, batch_dims=1)

        focal_term = (1 - probs) ** self.gamma
        loss_per_sample = focal_term * ce
        if self.alpha is not None:
            alpha = tf.gather(self.alpha, y_true)
            loss_per_sample *= alpha

        # Apply reduction
        if self.user_reduction == 'mean':
            loss = tf.reduce_mean(loss_per_sample)
        elif self.user_reduction == 'sum':
            loss = tf.reduce_sum(loss_per_sample)
        else:
            loss = loss_per_sample

        return loss

    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha': self.alpha.numpy().tolist() if self.alpha is not None else None,
            'gamma': self.gamma.numpy(),
            'reduction': self.user_reduction,
        })
        return config


class MacroCrossEntropy:
    """
    Computes the macro cross entropy (averaged cross-entropy per class) using only TensorFlow operations.
    """
    def __init__(self, from_logits=False):
        self.from_logits = from_logits
        self.ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE,
                                                                     from_logits=from_logits)

    @tf.function
    def __call__(self, y_true, y_pred):
        if len(y_true.shape) > 1:
            y_true = y_true[:, 0]
        ce = self.ce_loss(y_true, y_pred)
        unique_labels, new_idx = tf.unique(y_true)

        # Compute the mean loss for each unique label group
        seg_means = tf.math.unsorted_segment_mean(ce, new_idx, tf.shape(unique_labels)[0])

        # Average the group means to get the macro cross entropy
        macro_ce = tf.reduce_mean(seg_means)
        return macro_ce


@tf.keras.utils.register_keras_serializable(package='CustomMetrics')
class MacroCrossEntropyMetric(tf.keras.metrics.Metric):
    """
    A TensorFlow metric that computes the macro cross entropy across batches.
    It accumulates y_true and y_pred over all batches (using state variables),
    and computes the macro cross entropy on the complete accumulated data.

    Note: num_classes must be provided so that y_pred tensors can be concatenated.
    """
    def __init__(self, from_logits=False, num_classes=None, name='macro_cross_entropy', **kwargs):
        if num_classes is None:
            raise ValueError("You must provide num_classes to initialize MacroCrossEntropyMetric.")
        super(MacroCrossEntropyMetric, self).__init__(name=name, **kwargs)
        self.from_logits = from_logits
        self.num_classes = num_classes
        self.mce = MacroCrossEntropy(from_logits=from_logits)
        # Initialize state variables as empty tensors.
        # We use resource variables so their values persist across function graphs.
        self.y_true_accum = tf.Variable(
            initial_value=tf.constant([], dtype=tf.int32),
            trainable=False,  # not trainable
            shape=tf.TensorShape([None]),
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA
        )
        self.y_pred_accum = tf.Variable(
            initial_value=tf.zeros([0, num_classes], dtype=tf.float32),
            trainable=False,
            shape=tf.TensorShape([None, num_classes]),
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA
        )

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Reshape and cast the inputs to the appropriate type.
        y_true = tf.reshape(tf.cast(y_true, tf.int32), [-1])
        y_pred = tf.reshape(tf.cast(y_pred, tf.float32), (-1, self.num_classes))

        # Update state by concatenating the new batch data to the previously accumulated tensors.
        self.y_true_accum.assign(tf.concat([self.y_true_accum, y_true], axis=0))
        self.y_pred_accum.assign(tf.concat([self.y_pred_accum, y_pred], axis=0))

    @tf.function
    def result(self):
        return self.mce(self.y_true_accum, self.y_pred_accum)

    @tf.function
    def reset_state(self):
        self.y_true_accum.assign(tf.constant([], dtype=tf.int32))
        self.y_pred_accum.assign(tf.zeros([0, self.num_classes], dtype=tf.float32))

    def get_config(self):
        return {
            'from_logits': self.from_logits,
            'num_classes': self.num_classes,
            'name': self.name,
        }
