import numpy as np
from keras.models import Model, load_model

class Seq2SeqModelLoader:
    def __init__(self, path, latent_dim=256):
        self.model = load_model(path)
        encoder_inputs = model.input[0]
        decoder_inputs = model.input[1]
        encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output
        decoder_lstm = model.layers[3]
        decoder_dense = model.layers[4]
        encoder_states = [state_h_enc, state_c_enc]
        

        decoder_state_input_h = Input(shape=(latent_dim,), name='input_3')
        decoder_state_input_c = Input(shape=(latent_dim,), name='input_4')
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
            decoder_inputs, 
            initial_state=decoder_states_inputs
        )

        self.encoder_model = Model(
            encoder_inputs,
            encoder_states
        )
        self.decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_dense(decoder_outputs)] + [state_h_dec, state_c_dec]
        )
    
    def decode_sequence(input_seq, reverse_target_char_index):
        states_value = self.encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, target_token_index['\t']] = 1.
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + states_value)
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char
            if (sampled_char == '\n' or
            len(decoded_sentence) > max_decoder_seq_length):
                stop_condition = True
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.
            states_value = [h, c]
        return decoded_sentence