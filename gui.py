import gradio as gr
from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model('model.h5')
labels = ['no', 'yes']


def preprocessing_function(imgs):
    image = cv2.GaussianBlur(imgs, (5, 5), 0)
    image = image * 2
    image = image + 0.2
    return np.asarray(image)


def classify_image(inp):
    inp = preprocessing_function(inp)
    inp = inp.reshape((-1, 224, 224, 3))
    prediction = model.predict(inp).flatten()
    return {labels[i]: float(prediction[i]) for i in range(2)}


image = gr.inputs.Image(shape=(224, 224))
label = gr.outputs.Label(num_top_classes=2, label="PREDICTION")

gr.Interface(fn=classify_image, inputs=image, outputs=label,
             title="BRAIN TUMOR DETECTION").launch()
