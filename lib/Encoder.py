from keras.layers import Input, LSTM

class Encoder:
    def __init__(self, num_encoder_tokens, latent_dim=256):
        self.inputs = Input(shape=(None, num_encoder_tokens))
        outputs, state_h, state_c = LSTM(latent_dim, return_state=True)(self.inputs)
        self.states = [state_h, state_c]