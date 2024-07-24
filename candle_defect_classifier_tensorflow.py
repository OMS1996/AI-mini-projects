"""
Change the output layer to have the same number of neurons as the number of classes, with a softmax activation function.
Use the categorical cross-entropy loss function instead of binary cross-entropy.
Adjust the metric to categorical accuracy.

Assuming you have n_classes as the number of classes for classifying defects in candles, here is the modified code:

"""

import tensorflow as tf

def classify_defects(n_classes):
    model = tf.keras.models.Sequential([
        # 1. Convolutional layer with 16 filters, kernel size 3x3, ReLU activation, He normal initialization
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3), padding='valid',
                               kernel_initializer='he_normal'),
        # 2. Max pooling layer with size 2x2
        tf.keras.layers.MaxPooling2D((2, 2), padding='valid'),
        # 3. Convolutional layer with 32 filters, kernel size 3x3, ReLU activation, He normal initialization
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='valid', kernel_initializer='he_normal'),
        # 4. Max pooling layer with size 2x2
        tf.keras.layers.MaxPooling2D((2, 2), padding='valid'),
        # 5. Convolutional layer with 64 filters, kernel size 3x3, ReLU activation, He normal initialization
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='valid', kernel_initializer='he_normal'),
        # 6. Flatten layer
        tf.keras.layers.Flatten(),
        # 7. Dense layer with 20 neurons, ReLU activation, He normal initialization
        tf.keras.layers.Dense(20, activation='relu', kernel_initializer='he_normal'),
        # 8. Dense layer with n_classes neurons, softmax activation, Glorot normal initialization
        tf.keras.layers.Dense(n_classes, activation='softmax', kernel_initializer='glorot_normal')
    ])

    # Compile the model with Adam optimizer, learning rate 0.01, categorical cross-entropy loss, and categorical accuracy metric
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['categorical_accuracy']
    )
    
    return model

# Example usage
n_classes = 5  # Adjust this to the number of classes you have
model = classify_defects(n_classes)
model.summary()
