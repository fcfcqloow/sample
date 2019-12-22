from keras.models import Model
class Seq2Seq:
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder
        self.model =  Model([
                encoder.inputs,
                decoder.inputs
            ],
            decoder.outputs
        )
    
    def trainModel(
        self,
        encoder_input_data,
        decoder_input_data,
        decoder_target_data,
        epochs,
        batch_size,
        validation_split=0.2,
        optimizer='rmsprop',
        loss='categorical_crossentropy'
        ):

        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy'
        )
        self.model.fit(
            [
                encoder_input_data,
                decoder_input_data
            ],
            decoder_target_data,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split
        )
    
    def saveModel(self, path='seq2seq.h5'):
        self.model.save(path)
