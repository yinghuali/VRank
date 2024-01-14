# source activate py37
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.densenet import DenseNet121


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def build(self, input_shape):
        self.position_embeddings.build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'sequence_length': self.sequence_length,
            'output_dim': self.output_dim
        })

        return config

    def call(self, inputs):
        # The inputs are of shape: `(batch_size, frames, num_features)`
        inputs = tf.cast(inputs, self.compute_dtype)
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.3
        )
        self.dense_proj = tf.keras.Sequential(
            [
                layers.Dense(dense_dim, activation=tf.keras.activations.gelu),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'dense_dim': self.dense_dim,
            'num_heads': self.num_heads
        })

        return config


def build_feature_extractor():
    feature_extractor = DenseNet121(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(112, 112, 3),
    )
    preprocess_input = tf.keras.applications.densenet.preprocess_input

    inputs = tf.keras.Input((112, 112, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return tf.keras.Model(inputs, outputs, name="feature_extractor")


def get_vt_model(shape, num_classes):
    sequence_length = 60
    embed_dim = 1024
    dense_dim = 4
    num_heads = 1
    classes = num_classes

    inputs = tf.keras.Input(shape=shape)
    x = PositionalEmbedding(sequence_length, embed_dim, name="frame_position_embedding")(inputs)
    x = TransformerEncoder(embed_dim, dense_dim, num_heads, name="transformer_layer")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model


# # tensorflow 2.4.0
# num_classes = 11
# video_frames = np.zeros((50, 16, 112, 112, 3), dtype=np.float32)
# feature_extractor = build_feature_extractor()
# frame_features = [feature_extractor(frame) for frame in video_frames]
# frame_features = tf.stack(frame_features, axis=0)   # (50, 16, 1024)
# model = get_compiled_model(frame_features.shape[1:], num_classes)
# print(model.summary())