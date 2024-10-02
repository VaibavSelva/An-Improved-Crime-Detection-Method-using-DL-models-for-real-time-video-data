import cv2
import numpy as np
import tensorflow as tf
from playsound import playsound
mappings = {0: 'Abuse',
            1: 'Arson',
            2: 'Assault',
            3: 'Explosion',
            4: 'Fighting',
            5: 'Random',
            6: 'Shooting',
            7: 'Shoplifting',
            8: 'Vandalism'}


def load_model():
    pretrained = tf.keras.models.load_model(
        r"/Users/vaibavthalapathy/Documents/crime/Normal/model/TFVideoSwinS_K400_IN1K_P244_W877_32x224")
    pretrained.trainable = False
    ip1 = tf.keras.Input(shape=(32, 224, 224, 3))
    embed = pretrained(ip1)
    embed = tf.keras.layers.Dense(256, activation='relu')(embed)
    embed = tf.keras.layers.Dropout(0.4)(embed)
    op = tf.keras.layers.Dense(9, activation='softmax')(embed)
    model = tf.keras.Model(ip1, op)
    model.load_weights(r"/Users/vaibavthalapathy/Documents/crime/Normal/model_small.h5")
    return model


model = load_model()


def preprocess_frame(frame):
    frame = tf.image.resize(frame, (224, 224))
    frame = frame / 255
    return frame


frame_buffer = []
frame_count = 0

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_buffer.append(frame)
    frame_count += 1

    if frame_count == 32:
        input_data = np.array(frame_buffer)
        input_data = preprocess_frame(input_data)
        input_data = tf.expand_dims(input_data, 0)

        predictions = model(input_data)[0]
        predictions = mappings[tf.argmax(predictions).numpy()]
        if predictions in ["Abuse","Shooting","Explosion","Fighting"]:
            print(predictions)
            playsound("/Users/vaibavthalapathy/Documents/crime/mixkit-alert-alarm-1005.mp3")
        frame_buffer = []
        frame_count = 0

    # Show the frame
    cv2.imshow('frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
