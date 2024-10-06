import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ฟังก์ชันสำหรับคาดการณ์และสร้าง Confusion Matrix
def evaluate_model(test_data_dir):
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(48, 48),
        color_mode='grayscale',
        class_mode='sparse',
        batch_size=32,
        shuffle=False  # ไม่สับเปลี่ยนเพื่อให้สามารถจับคู่ผลลัพธ์ได้
    )

    # คาดการณ์
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # เอาค่าจริง
    y_true = test_generator.classes

    # สร้าง Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    
    # แสดงผล Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_generator.class_indices)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

model = tf.keras.models.load_model('Model/CK_model.h5')
test_data_dir = 'CKPlusTestDataset/'  # เปลี่ยนเป็นที่อยู่ของข้อมูลทดสอบ
evaluate_model(test_data_dir)
