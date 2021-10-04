import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.layers.core import Dropout


class Attention(tf.keras.Model):
    def __init__(self, units, embedding_dim, hidden_dim):
        super().__init__()
        # self.W1 = tf.keras.Dense(units)
        # self.W2 = tf.keras.Dense(units)
        # self.V = tf.keras.Dense(1)

        features = tf.keras.layers.Input(shape=(64, embedding_dim))
        context = tf.keras.layers.Input(shape=(1, hidden_dim))
        h1 = tf.keras.layers.Dense(units)(features)
        h2 = tf.keras.layers.Dense(units)(context)
        summed = tf.keras.layers.Add()([h1, h2])
        activated = tf.keras.activations.tanh(summed)
        activated = tf.keras.layers.Dropout(0.5)(activated)
        value = tf.keras.layers.Dense(1, activation='linear')(activated)
        self.model = tf.keras.Model([features, context], value)

    def call(self, inputs):
        features, hidden = inputs
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        attention_scores = self.model([features, hidden_with_time_axis])
        attention_weights = tf.nn.softmax(attention_scores, axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector


class Decoder(tf.keras.layers.Layer):
    def __init__(self, attention_hidden_units, hidden_state_size, embedding_dim, vocab_size):
        super().__init__()
        self.concatenate = tf.keras.layers.Concatenate()
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.decoder_h1 = tf.keras.layers.Dense(
            hidden_state_size, activation='relu')
        self.decoder_out = tf.keras.layers.Dense(vocab_size)
        self.lstmCell = tf.keras.layers.LSTMCell(hidden_state_size, recurrent_dropout=0.5)
        self.state_size = self.lstmCell.state_size

    def call(self, inputs, state):
        context, embed_text = inputs
        concatenated = self.concatenate([embed_text, context])
        concatenated = self.dropout(concatenated)
        x, state = self.lstmCell(concatenated, state)
        h1 = self.decoder_h1(x)
        out = self.decoder_out(h1)

        return out, state


class ImageCaptioningCell(tf.keras.layers.Layer):
    def __init__(self, attention_hidden_units, hidden_state_size, embedding_dim, vocab_size):
        super().__init__()

        self.attention_model = Attention(
            attention_hidden_units, embedding_dim, hidden_state_size)
        self.decoder = Decoder(attention_hidden_units,
                               hidden_state_size, embedding_dim, vocab_size)
        # self.decoder = tf.keras.layers.RNN(
        #     self.decoder, return_sequences=True, return_state=True)

        self.hidden_state_size = hidden_state_size
        self.embedding_dim = embedding_dim
        self.state_size = [
            tf.TensorShape([self.hidden_state_size]),
            tf.TensorShape([self.hidden_state_size]),
            tf.TensorShape([64, self.embedding_dim])
        ]
    # def build(self, input_shape):
    #     batch_size = input_shape[0]
    #     self.state_size = (
    #         (batch_size, self.hidden_state_size),
    #         (batch_size, self.hidden_state_size),
    #         (batch_size, 64, self.embedding_dim)
    #     )
    #     self.built = True

    def call(self, inputs, states):
        h, c, image_features = states
        # image_features = self.encoder(image)
        context = self.attention_model([image_features, h])
        out, [new_h, new_c] = self.decoder(
            [context, inputs], state=[h, c])
        return out, [new_h, new_c, image_features]


class ImageCaptioning(tf.keras.Model):
    def __init__(self, attention_hidden_units, hidden_state_size, embedding_dim, vocab_size, input_shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = tf.keras.models.Sequential([
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(embedding_dim, input_shape=input_shape)
        ])
        self.init_c = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(hidden_state_size, activation='linear'),
        ])
        self.init_h = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(hidden_state_size, activation='linear'),
        ])
        self.cell = ImageCaptioningCell(
            attention_hidden_units, hidden_state_size, embedding_dim, vocab_size)
        self.rnn = tf.keras.layers.RNN(self.cell, return_sequences=True)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    def call(self, inputs):
        image, text = inputs
        embed_text = self.embedding(text)
        image_features = self.encoder(image)
        mean_features = tf.reduce_mean(image_features, axis=1)
        init_h = self.init_h(mean_features)
        init_c = self.init_c(mean_features)

        return self.rnn(embed_text, initial_state=[init_h, init_c, image_features])

    def get_rnn(self, image):
        image_features = self.encoder(image)
        text = tf.keras.layers.Input(batch_shape=[1, None])
        embed_text = self.embedding(text)
        mean_features = tf.reduce_mean(image_features, axis=1)
        init_h = self.init_h(mean_features)
        init_c = self.init_c(mean_features)
        rnn = tf.keras.layers.RNN(self.cell, return_sequences=True, stateful=True)
        out = rnn(embed_text, initial_state=(init_h, init_c, image_features))
        return tf.keras.models.Model(text, out)


