# pip install numpy tensorflow matplotlib pillow

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# ฟังก์ชันสำหรับเตรียมภาพ
def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(48, 48), color_mode='grayscale')
    img = image.img_to_array(img) / 255.0  # ทำ Normalization
    img = np.expand_dims(img, axis=0)  # เพิ่มมิติของ array ตามความต้องการของ CNN
    return img

# Load Model ที่เคยเทรนด์ไว้
model = tf.keras.models.load_model('Model/CK_model.h5')

def predict_emotion(img_path):
    img = prepare_image(img_path)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    
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
    
    predicted_emotion = emotion_labels[predicted_class]
    
    # แสดงภาพต้นฉบับและภาพที่เตรียมแล้ว
    plt.figure(figsize=(10, 5))
    
    # ภาพที่ 1: ต้นฉบับ
    plt.subplot(1, 2, 1)
    plt.imshow(image.load_img(img_path, color_mode='rgb'))
    plt.axis('off')
    plt.title('Original Image')  # แสดงคำอธิบาย
    
    # ภาพที่ 2: หลัง prepare_image
    plt.subplot(1, 2, 2)
    plt.imshow(img.squeeze(), cmap='gray')  # ใช้ squeeze และ cmap='gray' สำหรับภาพ grayscale
    plt.axis('off')  # ปิดแกน
    plt.title(f'Predicted Emotion: {predicted_emotion}')  # แสดงชื่ออารมณ์ที่ทำนาย
    
    plt.show()  # แสดงภาพ
    
    # พิมพ์ Output vector 
    print(predictions)
    
    return predicted_emotion

# ใช้งาน
# เปลี่ยนเป็นที่อยู่ของภาพที่ต้องการทำนาย
img_paths = ['TestData/ch_1.png', 'TestData/ch_2.png', 'TestData/ch_3.png', 'TestData/new.png']
for path in img_paths:
    predicted_emotion = predict_emotion(path)
    print(f'Predicted emotion class: {predicted_emotion}')
