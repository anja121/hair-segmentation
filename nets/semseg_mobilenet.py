from tensorflow.keras import layers, Model
from tensorflow.keras.activations import sigmoid


def _padding_layer(inputs, strides=(1, 1)):
    if strides == (1, 1):
        return inputs, "same"
    else:
        return layers.ZeroPadding2D(padding=(1, 1))(inputs), "valid"


def _depthwise_conv_block(inputs, depth_multiplier=1, strides=(1, 1)):
    x, padding_type = _padding_layer(inputs=inputs, strides=strides)
    x = layers.DepthwiseConv2D(kernel_size=(3, 3),
                               padding=padding_type,
                               depth_multiplier=depth_multiplier,
                               strides=strides,
                               use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    return x


def _pointwise_conv_block(inputs, filters):
    x = layers.Conv2D(filters=filters,
                      kernel_size=(1, 1),
                      strides=(1, 1),
                      padding="same",
                      use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    return x


def depthwise_separable_conv_block_encoder(inputs, pointwise_filters, depth_multiplier=1, depthwise_strides=(1, 1)):

    x = _depthwise_conv_block(inputs, depth_multiplier, depthwise_strides)
    x = _pointwise_conv_block(x, pointwise_filters)
    return x


def upsampling_block(inputs):
    return layers.UpSampling2D(size=(2, 2))(inputs)


def depthwise_separable_conv_block_decoder(inputs, pointwise_filters, depth_multiplier=1, depthwise_strides=(1, 1)):
    x, padding_type = _padding_layer(inputs=inputs, strides=depthwise_strides)
    x = layers.DepthwiseConv2D(kernel_size=(3, 3),
                               padding=padding_type,
                               depth_multiplier=depth_multiplier,
                               strides=depthwise_strides,
                               use_bias=False)(x)
    x = layers.Conv2D(filters=pointwise_filters,
                      kernel_size=(1, 1),
                      strides=(1, 1),
                      padding="same",
                      use_bias=False)(x)

    x = layers.ReLU()(x)

    return x


def merge_layers(layers_arr, merge_type):
    if merge_type == "concat":
        return layers.concatenate(layers_arr)
    elif merge_type == "add":
        return layers.add(layers_arr)
    else:
        raise Exception("Given merge type is not implemented!")


def build_model(shape=(224, 224, 3), merge_type="concat"):
    inputs = layers.Input(shape=shape)

    # input

    # conv 3x3
    x = layers.ZeroPadding2D(padding=(1, 1))(inputs)
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), padding="valid", use_bias=False, strides=(2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Encoder part

    # depthwise conv 3x3 + BatchNorml + ReLu + Conv 1x1 + BatchNorm + Relu blocks
    skip_connection_0 = depthwise_separable_conv_block_encoder(inputs=x,
                                                               pointwise_filters=64)
    x = depthwise_separable_conv_block_encoder(inputs=skip_connection_0,
                                               pointwise_filters=128,
                                               depthwise_strides=(2, 2))
    skip_connection_1 = depthwise_separable_conv_block_encoder(inputs=x,
                                                               pointwise_filters=128)
    x = depthwise_separable_conv_block_encoder(inputs=skip_connection_1,
                                               pointwise_filters=256,
                                               depthwise_strides=(2, 2))
    skip_connection_2 = depthwise_separable_conv_block_encoder(inputs=x,
                                                               pointwise_filters=256)

    x = depthwise_separable_conv_block_encoder(inputs=skip_connection_2,
                                               pointwise_filters=512,
                                               depthwise_strides=(2, 2))
    for i in range(4):
        x = depthwise_separable_conv_block_encoder(inputs=x,
                                                   pointwise_filters=512)

    skip_connection_3 = depthwise_separable_conv_block_encoder(inputs=x,
                                                               pointwise_filters=512)
    x = depthwise_separable_conv_block_encoder(inputs=skip_connection_3,
                                               pointwise_filters=1024,
                                               depthwise_strides=(2, 2))
    x = depthwise_separable_conv_block_encoder(inputs=x,
                                               pointwise_filters=1024)

    # Decoder part

    # Upsampling
    # merging skip connection layers
    # depthwise conv 3x3 + Conv 1x1  + Relu blocks

    x = upsampling_block(x)
    x = merge_layers([x, skip_connection_3], merge_type)
    x = depthwise_separable_conv_block_decoder(inputs=x,
                                               pointwise_filters=64)
    x = upsampling_block(x)
    x = merge_layers([x, skip_connection_2], merge_type)
    x = depthwise_separable_conv_block_decoder(inputs=x,
                                               pointwise_filters=64)

    x = upsampling_block(x)
    x = merge_layers([x, skip_connection_1], merge_type)
    x = depthwise_separable_conv_block_decoder(inputs=x,
                                               pointwise_filters=64)

    x = upsampling_block(x)
    x = merge_layers([x, skip_connection_0], merge_type)
    x = depthwise_separable_conv_block_decoder(inputs=x,
                                               pointwise_filters=64)

    x = upsampling_block(x)
    x = depthwise_separable_conv_block_decoder(inputs=x,
                                               pointwise_filters=64)

    # output
    # conv 1x1 + Sigmoid
    x = layers.Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1))(x)
    outputs = sigmoid(x)

    return Model(inputs, outputs)




