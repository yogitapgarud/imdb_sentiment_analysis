import argparse
import time

import tensorflow as tf
import numpy as np

import build_model
import load_dataset
import vectorize_data
import explore_data

FLAGS = None

def train_ngram_model(data,
                      learning_rate=1e-3,
                      epochs=1000,
                      batch_size=128,
                      layers=2,
                      units=64,
                      dropout_rate=0.2):

    (train_texts, train_labels), (val_texts, val_labels) = data

    num_classes = explore_data.get_num_classes(train_labels)
    unexpected_labels = [v for v in val_labels if v not in range(num_classes)]
    if len(unexpected_labels):
        raise ValueError('Unexpected label values found in validation set')

    x_train, x_val = vectorize_data.ngram_vectorize(train_texts, train_labels, val_texts)

    model = build_model.mlp_model(layers=layers,
                                  units=units,
                                  dropout_rate=dropout_rate,
                                  input_shape=x_train.shape[1:],
                                  num_classes=num_classes)

    if num_classes == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'sparse_categorical_crossentropy'
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)]

    history = model.fit(
            x_train,
            train_labels,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=(x_val, val_labels),
            verbose=2,
            batch_size=batch_size)

    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
            acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

    model.save('IMDb_mlp_model.h5')
    return history['val_acc'][-1], history['val_loss'][-1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='.',
                        help='input data directory')
    FLAGS, unparsed = parser.parse_known_args()

    # Using the IMDb movie reviews dataset to demonstrate training n-gram model
    data = load_dataset.load_imdb_sentiment_analysis_dataset(FLAGS.data_dir)
    train_ngram_model(data)