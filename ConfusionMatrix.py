import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# โหลดโมเดลที่ฝึกเสร็จแล้ว
model = tf.keras.models.load_model('Model/model.h5')

# สร้าง ImageDataGenerator สำหรับการโหลดข้อมูลทดสอบ
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    'Dataset_split/test/',  # พาธที่เก็บโฟลเดอร์ของข้อมูลทดสอบ
    target_size=(48, 48),
    color_mode='grayscale',
    class_mode='sparse',
    batch_size=32,
    shuffle=False  # ไม่ต้องการให้ข้อมูลสับเปลี่ยนในระหว่างการทดสอบ
)

# ทำนายผลลัพธ์จากโมเดล
predictions = model.predict(test_generator, steps=len(test_generator), verbose=1)

# แปลงค่าการทำนายให้เป็นคลาส (จาก probabilistic output ให้เป็นคลาสที่มีความน่าจะเป็นสูงสุด)
predicted_classes = np.argmax(predictions, axis=1)

# แสดง Confusion Matrix
true_classes = test_generator.classes

# คำนวณ confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)

# แสดง confusion matrix ด้วย heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_generator.class_indices, yticklabels=test_generator.class_indices)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
