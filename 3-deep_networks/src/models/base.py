from typing import Tuple

from tensorflow.keras import Model
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input


def basemodel(shape: Tuple) -> Model:
    """input shape, output a Dense model with one hidden layer

    Args:
        shape (Tuple): datashape, excluding batchsize

    Returns:
        Model: A keras model
    """
    input = Input(shape=shape)
    x = Dense(30, activation="relu")(input)
    output = Dense(1)(x)

    model = Model(inputs=[input], outputs=[output])
    return model


def basedeep_model(shape: Tuple) -> Model:
    """input shape, output a Dense model with one hidden layer

    Args:
        shape (Tuple): datashape, excluding batchsize

    Returns:
        Model: A keras model
    """
    input = Input(shape=shape)
    x = Dense(30, activation="relu")(input)
    x = Dense(30, activation="relu")(x)
    x = Dense(30, activation="relu")(x)
    output = Dense(1)(x)

    model = Model(inputs=[input], outputs=[output])
    return model


def double_input(shape: Tuple) -> Model:
    """
        Create a model that has two inputs, one input flows through a hidden
        layer with 50 units and gets concatenated with the second input.

        The motivation for this could be, that you have one part of the input
        that is expected to have very complex relationships to the output and
        thus needs and hidden layer and 50 units.
        The second part of the input is very basic, and will work with a much
        simpler model. This way, the second part is not touched by the hidden
        layer, and only used in the last layer.


    - the concatenated layers are fed into a Dense layer with a single unit, which is the output of the model.

        Args:
            shape (Tuple): specification of the shape, excluding batch

        Returns:
            Model: an tf.keras.Model
    """
    # two lines of code:
    # two Input layers (`inputa` and `inputb`), both for data with shape (batch
    #   x shape).
    inputa = Input(shape=shape)
    inputb = Input(shape=shape)

    # one line of code: inputa feeds into a Dense layer with 50 units
    x1 = Dense(50, activation="relu")(inputa)

    # one line of code: use tf.concat over dimension 1 (so, tf.concat([], 1))
    # and concatenate x1 and inputb.
    x2 = tf.concat([x1, inputb], 1)

    output = Dense(1)(x2)

    model = Model(inputs=[inputa, inputb], outputs=[output])
    return model
