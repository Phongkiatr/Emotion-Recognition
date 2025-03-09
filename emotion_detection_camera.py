import cv2
import numpy as np
import tensorflow as tf
import threading
from tensorflow.keras.preprocessing import image
import os
import time

# ฟังก์ชันสำหรับเตรียมภาพ
def prepare_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # แปลงเป็น Grayscale
    img = cv2.resize(img, (48, 48))  # ปรับขนาดภาพให้ตรงกับโมเดล
    img = np.expand_dims(img, axis=-1)  # เพิ่มมิติของสี (1 ช่องสำหรับ grayscale)
    img = np.expand_dims(img, axis=0)  # เพิ่มมิติ batch ให้เป็น (1, 48, 48, 1)
    img = img / 255.0  # Normalization
    return img

# โหลดโมเดลที่เคยเทรนด์ไว้
model = tf.keras.models.load_model('Model/model.h5', compile=False)

# แปลงหมายเลขคลาสเป็นชื่ออารมณ์
emotion_labels = {    
    0: 'happy',   
    1: 'sad',       
    2: 'surprise'   
}

# ฟังก์ชันทำนายอารมณ์
def predict_emotion(face):
    img = prepare_image(face)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    predicted_emotion = emotion_labels[predicted_class]
    return predicted_emotion

def capture_and_predict(cap, stop_event):
    # สร้างโฟลเดอร์สำหรับบันทึกภาพ (ถ้ายังไม่มี)
    if not os.path.exists('saved_images'):
        os.makedirs('saved_images')

    # ตัวแปรสำหรับการถ่ายภาพ
    image_counter = 0
    last_predicted_emotion = None  # เก็บอารมณ์ที่ทำนายไว้ครั้งก่อน
    countdown = 0  # ตัวแปรนับถอยหลัง
    countdown_start_time = 0  # ตัวแปรเก็บเวลาเริ่มต้นการนับถอยหลัง
    frame_counter = 0  # ตัวแปรนับจำนวนเฟรม

    while not stop_event.is_set():  # ใช้ stop_event สำหรับการหยุด
        # อ่านภาพจากกล้อง
        ret, frame = cap.read()  # อ่านภาพจากกล้อง
        if not ret:
            break

        # ทำการสะท้อนภาพ (Mirror)
        frame = cv2.flip(frame, 1)  # Flip ในแนวนอน

        # ตรวจจับใบหน้า
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

        # คำนวณการทำนายทุกๆ n เฟรม
        if frame_counter % 1 == 0:
            # ทำนายอารมณ์สำหรับแต่ละใบหน้า
            for (x, y, w, h) in faces:
                # ตัดเฉพาะส่วนใบหน้า
                face = frame[y:y+h, x:x+w]
                
                # ใช้ฟังก์ชันที่ทำนายอารมณ์
                predicted_emotion = predict_emotion(face)

                # ถ้าไม่ได้อยู่ในช่วงนับถอยหลัง
                if countdown <= 0:
                    # วาดกรอบและแสดงชื่ออารมณ์
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(frame, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                # ถ้าอารมณ์ที่ทำนายคือ "happy" และยังไม่เคยถ่ายภาพไปแล้ว
                if predicted_emotion == 'happy' and last_predicted_emotion != 'happy':
                    countdown = 1  # เริ่มนับถอยหลัง 3 วินาที
                    countdown_start_time = time.time()  # เก็บเวลาเริ่มต้น

                if countdown > 0:
                    # คำนวณเวลาที่เหลือจากเวลาปัจจุบัน
                    elapsed_time = time.time() - countdown_start_time
                    remaining_time = int(countdown - elapsed_time)

                    if remaining_time <= 0:
                        # บันทึกภาพเมื่อถึงเวลา
                        image_counter += 1
                        image_filename = f'saved_images/happy_{image_counter}.jpg'
                        cv2.imwrite(image_filename, frame)  # บันทึกภาพลงไฟล์
                        print(f'Image saved as {image_filename}')
                        countdown = -1  # หยุดการนับถอยหลัง

                # อัปเดตอารมณ์ล่าสุด
                last_predicted_emotion = predicted_emotion

        frame_counter += 1

        # แสดงภาพ
        cv2.imshow('Emotion Recognition', frame)

        # กด 'q' หรือปิดหน้าต่างเพื่อออกจากโปรแกรม
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    cap.release()
    cv2.destroyAllWindows()


# เปิดกล้อง
cap = cv2.VideoCapture(0)

# ตั้งค่าความละเอียดที่ต้องการ (เช่น 1920x1080)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # ความกว้างของเฟรม
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # ความสูงของเฟรม

# ตั้งค่าการจับภาพในอัตรา FPS ที่เป็นไปตามกล้อง (ไม่บังคับ FPS สูงเกินไป)
cap.set(cv2.CAP_PROP_FPS, 60)  # ใช้ FPS ที่กล้องรองรับ

# ตรวจสอบความละเอียดที่ตั้งไว้
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Resolution: {frame_width}x{frame_height}")

# สร้าง event สำหรับการหยุดการทำงาน
stop_event = threading.Event()

# สร้างเทรดแยกสำหรับการจับภาพและทำนาย
capture_thread = threading.Thread(target=capture_and_predict, args=(cap, stop_event))
capture_thread.start()

capture_thread.join()  # รอให้เทรดเสร็จสิ้นก่อนปิดโปรแกรม
