import cv2
import os
import time
from model_prediction import predict_emotion

def capture_and_predict(cap):
    if not os.path.exists('saved_images'):
        os.makedirs('saved_images')

    image_counter = 0
    last_predicted_emotion = None
    countdown = 0
    countdown_start_time = 0
    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

        if frame_counter % 2 == 0:
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                predicted_emotion = predict_emotion(face)

                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                if predicted_emotion == 'happy' and last_predicted_emotion != 'happy':
                    countdown = 3
                    countdown_start_time = time.time()

                if countdown > 0:
                    elapsed_time = time.time() - countdown_start_time
                    remaining_time = int(countdown - elapsed_time)

                    if remaining_time > 0:
                        cv2.putText(frame, f'Capturing in {remaining_time}...', (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    elif remaining_time <= 0:
                        image_counter += 1
                        image_filename = f'saved_images/happy_{image_counter}.jpg'
                        cv2.imwrite(image_filename, frame)
                        print(f'Image saved as {image_filename}')
                        countdown = -1

                last_predicted_emotion = predicted_emotion

        frame_counter += 1
        cv2.imshow('Emotion Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
