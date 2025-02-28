import numpy as np
import tensorflow as tf
from image_processing import prepare_image

# โหลดโมเดลที่เคยเทรนไว้
model = tf.keras.models.load_model('Model/CK_model.h5', compile=False)

# แปลงหมายเลขคลาสเป็นชื่ออารมณ์
emotion_labels = {
    0: 'angry',    
    1: 'contempt',   
    2: 'disgust',     
    3: 'fear',     
    4: 'happy',   
    5: 'sad',       
    6: 'surprise'   
}

# ฟังก์ชันทำนายอารมณ์
def predict_emotion(face):
    img = prepare_image(face)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    predicted_emotion = emotion_labels[predicted_class]
    return predicted_emotion
