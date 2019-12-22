# https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py

from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.utils import plot_model
import numpy as np

from Encoder import *
from Decoder import *
from train import *
from Seq2SeqModelLoader import *

num_samples = 20000 
data_path = 'sample.txt'
 
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text = line.split('\t')
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)
input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.

print(
    num_encoder_tokens,
    num_decoder_tokens,
    encoder_input_data,
    decoder_input_data,
    decoder_target_data
)
encoder = Encoder(num_encoder_tokens)
decoder = Decoder(num_decoder_tokens, encoder)
seq2seq = Seq2Seq(encoder, decoder)
seq2seq.model.summary()
seq2seq.trainModel(
    encoder_input_data,
    decoder_input_data,
    decoder_target_data,
    30, # epochs
    64  # batch size
)
seq2seq.saveModel(path='model.h5')


reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())
model = Seq2SeqModelLoader('model.h5')


def main():
    for seq_index in range(10):
        input_seq = encoder_input_data[seq_index: seq_index + 1]
        decoded_sentence = model.decode_sequence(input_seq, reverse_target_char_index)
        print('-')
        print('Input sentence:', input_texts[seq_index])
        print('Decoded sentence:', decoded_sentence)

if __name__ == "__main__":
    main() 
 
