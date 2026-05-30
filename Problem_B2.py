# =============================================================================
# PROBLEM B2
#
# Build a classifier for the Fashion MNIST dataset.
# The test will expect it to classify 10 classes.
# The input shape should be 28x28 monochrome. Do not resize the data.
# Your input layer should accept (28, 28) as the input shape.
#
# Don't use lambda layers in your model.
#
# Desired accuracy AND validation_accuracy > 83%
# =============================================================================

import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        # Set a safe target at 84.5% (0.845) to confidently exceed the 83% threshold
        if(logs.get('accuracy') > 0.845 and logs.get('val_accuracy') > 0.845):
            print("\nTarget accuracy > 84.5% reached! Stopping training...")
            self.model.stop_training = True

def solution_B2():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    
    # Load the dataset (split into training and test sets)
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # NORMALIZE YOUR IMAGE HERE
    # Scale pixel values from 0-255 to the range 0.0 - 1.0
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # DEFINE YOUR MODEL HERE
    model = tf.keras.models.Sequential([
        # Input layer must accept (28, 28) and be flattened to a 1D array
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        # Hidden layer to learn clothing patterns
        tf.keras.layers.Dense(128, activation='relu'),
        # End with 10 Neuron Dense, activated by softmax
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # COMPILE MODEL HERE
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Initialize callback
    callbacks = myCallback()

    # TRAIN YOUR MODEL HERE
    # Use validation_data to obtain val_accuracy
    model.fit(
        x_train, 
        y_train, 
        epochs=15, 
        validation_data=(x_test, y_test), 
        callbacks=[callbacks], 
        verbose=1
    )

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B2()
    model.save("model_B2.h5")