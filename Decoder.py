from keras.layers import Input, LSTM, Dense
 
class Decoder:
    def __init__(self, num_decoder_tokens, encoder, latent_dim=256, activation='softmax'):
        self.inputs = Input(shape=(None, num_decoder_tokens))
        outputs, _, _ = LSTM(latent_dim, return_sequences=True, return_state=True)(self.inputs, initial_state=encoder.states)
        self.outputs = Dense(num_decoder_tokens, activation=activation)(outputs)