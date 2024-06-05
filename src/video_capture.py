# import the opencv library
import cv2
import tensorflow as tf
import numpy as np
import os

# Define model path
model_path = "model/model.h5"

# Check if the model exists in the directory
if not os.path.exists(model_path):
    import gdown.download
    model_drive_id = "1S_whLRK6NzAomr6Mwcc19uZsfF-ozqgz"
    gdown.download(f"https://drive.google.com/uc?id={model_drive_id}", model_path, quiet=False)

# define a video capture object
vid = cv2.VideoCapture(0)

# load keras model
model = tf.keras.models.load_model(model_path, compile=False)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

label_list = np.array(['biological', 'cardboard', 'clothes',
                       'glass', 'metal', 'paper', 'plastic', 'trash'])

while True:

    # Capture the video frame by frame
    ret, frame = vid.read()

    # Preprocess image
    input_shape = (224, 224)
    preprocessed_img = cv2.resize(frame, input_shape, interpolation=cv2.INTER_AREA) / 255.
    preprocessed_img = np.expand_dims(preprocessed_img, 0)

    # Predict the processed frame
    predicted = model.predict(preprocessed_img)[0]

    # Get the top k prediction output
    max_k = 3
    values, indices = tf.math.top_k(predicted, k=max_k)
    print(values)
    print(indices)

    # Display the top k prediction
    for i in range(max_k):
        text = f"{label_list[indices[i]]}: {values[i]}"
        coordinates = (10, 40*(i+1))
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 0, 255)
        thickness = 2

        # Display the resulting frame
        frame = cv2.putText(frame, text, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    # Break the video feeds
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
