import sys
import numpy as np
import pandas as pd

from lib.Encoder import *
from lib.Decoder import *
from lib.Seq2Seq import *

path = sys.argv[1]

def main():
    global num_encoder_tokens, num_decoder_tokens, encoder_input_data, decoder_input_data, decoder_target_data
    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    for line in lines[: len(lines) - 1]:
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
    
    # reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
    reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())
    pd.to_pickle(reverse_target_char_index, 'index.pkl')

def train(path, epochs=10, batch=32):
    encoder = Encoder(num_encoder_tokens)
    decoder = Decoder(num_decoder_tokens, encoder)
    seq2seq = Seq2Seq(encoder, decoder)
    seq2seq.model.summary()
    seq2seq.trainModel(
        encoder_input_data,
        decoder_input_data,
        decoder_target_data,
        epochs,
        batch
    )
    seq2seq.saveModel(path=path)

def addTrain(path, epochs=10, batch=32):

    
if __name__ == "__main__":
    main()