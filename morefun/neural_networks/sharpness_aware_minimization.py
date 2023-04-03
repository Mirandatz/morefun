# Since we can't update Tensorflow: https://github.com/keras-team/keras-cv/issues/291
# And we want to use SAM, we just copy-pasted its implementation
# https://github.com/keras-team/keras/blob/v2.12.0/keras/models/sharpness_aware_minimization.py#L31-L191

# isort: off
import copy

# import tensorflow.compat.v2 as tf
import tensorflow as tf

from keras.engine import data_adapter
from keras.layers import deserialize as deserialize_layer
from keras.models import Model

# from keras.saving.legacy.serialization import serialize_keras_object
serialize_keras_object = tf.keras.utils.serialize_keras_object

# from keras.saving.object_registration import register_keras_serializable
register_keras_serializable = tf.keras.utils.register_keras_serializable

from tensorflow.python.util.tf_export import keras_export  # noqa

# isort: on


@register_keras_serializable()
@keras_export("keras.models.experimental.SharpnessAwareMinimization", v1=[])
class SharpnessAwareMinimization(Model):  # type: ignore
    """Sharpness aware minimization (SAM) training flow.
    Sharpness-aware minimization (SAM) is a technique that improves the model
    generalization and provides robustness to label noise. Mini-batch splitting
    is proven to improve the SAM's performance, so users can control how mini
    batches are split via setting the `num_batch_splits` argument.
    Args:
      model: `tf.keras.Model` instance. The inner model that does the
        forward-backward pass.
      rho: float, defaults to 0.05. The gradients scaling factor.
      num_batch_splits: int, defaults to None. The number of mini batches to
        split into from each data batch. If None, batches are not split into
        sub-batches.
      name: string, defaults to None. The name of the SAM model.
    Reference:
      [Pierre Foret et al., 2020](https://arxiv.org/abs/2010.01412)
    """

    def __init__(self, model, rho=0.05, num_batch_splits=None, name=None):  # type: ignore
        super().__init__(name=name)
        self.model = model
        self.rho = rho
        self.num_batch_splits = num_batch_splits

    def train_step(self, data):  # type: ignore
        """The logic of one SAM training step.
        Args:
          data: A nested structure of `Tensor`s. It should be of structure
            (x, y, sample_weight) or (x, y).
        Returns:
          A dict mapping metric names to running average values.
        """
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        if self.num_batch_splits is not None:
            x_split = tf.split(x, self.num_batch_splits)
            y_split = tf.split(y, self.num_batch_splits)
        else:
            x_split = [x]
            y_split = [y]

        gradients_all_batches = []  # type: ignore
        pred_all_batches = []
        for x_batch, y_batch in zip(x_split, y_split):
            epsilon_w_cache = []
            with tf.GradientTape() as tape:
                pred = self.model(x_batch)
                loss = self.compiled_loss(y_batch, pred)
            pred_all_batches.append(pred)
            trainable_variables = self.model.trainable_variables
            gradients = tape.gradient(loss, trainable_variables)

            gradients_order2_norm = self._gradients_order2_norm(gradients)  # type: ignore
            scale = self.rho / (gradients_order2_norm + 1e-12)

            for gradient, variable in zip(gradients, trainable_variables):
                epsilon_w = gradient * scale
                self._distributed_apply_epsilon_w(
                    variable, epsilon_w, tf.distribute.get_strategy()
                )  # type: ignore
                epsilon_w_cache.append(epsilon_w)

            with tf.GradientTape() as tape:
                pred = self(x_batch)
                loss = self.compiled_loss(y_batch, pred)
            gradients = tape.gradient(loss, trainable_variables)
            if len(gradients_all_batches) == 0:
                for gradient in gradients:
                    gradients_all_batches.append([gradient])
            else:
                for gradient, gradient_all_batches in zip(
                    gradients, gradients_all_batches
                ):
                    gradient_all_batches.append(gradient)
            for variable, epsilon_w in zip(trainable_variables, epsilon_w_cache):
                # Restore the variable to its original value before
                # `apply_gradients()`.
                self._distributed_apply_epsilon_w(
                    variable, -epsilon_w, tf.distribute.get_strategy()
                )  # type: ignore

        gradients = []
        for gradient_all_batches in gradients_all_batches:
            gradients.append(tf.reduce_sum(gradient_all_batches, axis=0))
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        pred = tf.concat(pred_all_batches, axis=0)
        self.compiled_metrics.update_state(y, pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs):  # type: ignore
        """Forward pass of SAM.
        SAM delegates the forward pass call to the wrapped model.
        Args:
          inputs: Tensor. The model inputs.
        Returns:
          A Tensor, the outputs of the wrapped model for given `inputs`.
        """
        return self.model(inputs)

    def get_config(self):  # type: ignore
        config = super().get_config()
        config.update(
            {
                "model": serialize_keras_object(self.model),
                "rho": self.rho,
            }
        )
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):  # type: ignore
        # Avoid mutating the input dict.
        config = copy.deepcopy(config)
        model = deserialize_layer(config.pop("model"), custom_objects=custom_objects)
        config["model"] = model
        return super().from_config(config, custom_objects)

    def _distributed_apply_epsilon_w(self, var, epsilon_w, strategy):  # type: ignore
        # Helper function to apply epsilon_w on model variables.
        if isinstance(
            tf.distribute.get_strategy(),
            (
                tf.distribute.experimental.ParameterServerStrategy,
                tf.distribute.experimental.CentralStorageStrategy,
            ),
        ):
            # Under PSS and CSS, the AggregatingVariable has to be kept in sync.
            def distribute_apply(strategy, var, epsilon_w):  # type: ignore
                strategy.extended.update(
                    var,
                    lambda x, y: x.assign_add(y),
                    args=(epsilon_w,),
                    group=False,
                )

            tf.__internal__.distribute.interim.maybe_merge_call(
                distribute_apply, tf.distribute.get_strategy(), var, epsilon_w
            )
        else:
            var.assign_add(epsilon_w)

    def _gradients_order2_norm(self, gradients):  # type: ignore
        norm = tf.norm(
            tf.stack([tf.norm(grad) for grad in gradients if grad is not None])
        )
        return norm