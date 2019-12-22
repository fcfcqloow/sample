import pandas as pd
from lib.Seq2SeqModelLoader import *

model = Seq2SeqModelLoader(sys.argv[1])
reverse_target_char_index = pd.read_pickle("index.pkl")
while True:
    seq = input()
    decoded_sentence = model.decode_sequence(seq, reverse_target_char_index)
    print('Decoded sentence:', decoded_sentence)