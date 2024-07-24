import tensorflow as tf


def classify_trucks():
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
        # 8. Dense layer with 1 neuron, sigmoid activation, Glorot normal initialization
        tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer='glorot_normal')
    ])

    # Compile the model with Adam optimizer, learning rate 0.01, binary cross-entropy loss, and binary accuracy metric
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['binary_accuracy']
    )
    
    return model
