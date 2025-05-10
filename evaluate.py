import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import pandas as pd

# โหลดโมเดลที่ฝึกเสร็จแล้ว
model = load_model('Model/model.h5')

# โหลดข้อมูล Test Set
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)  # ปรับค่าให้เป็น 0-1
test_generator = test_datagen.flow_from_directory(
    'Dataset_Split/test/',  # พาธไปยังโฟลเดอร์ทดสอบ
    target_size=(48, 48),
    color_mode="grayscale",
    class_mode="sparse",
    batch_size=32,
    shuffle=False  # ไม่ให้มีการสลับลำดับ
)

# ประเมินผลลัพธ์ของโมเดล
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# ทำนายผลลัพธ์
y_pred_prob = model.predict(test_generator, verbose=1)  # ได้ผลลัพธ์เป็นค่าความน่าจะเป็น
y_pred_classes = np.argmax(y_pred_prob, axis=1)  # แปลงเป็นคลาสที่มีค่าความน่าจะเป็นสูงสุด

# ค่าจริง (True labels)
y_true = test_generator.classes

# คำนวณ Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
print("\nConfusion Matrix:")
print(conf_matrix)

# คำนวณ Precision, Recall, F1-score
report = classification_report(y_true, y_pred_classes, target_names=test_generator.class_indices.keys())
print("\nClassification Report:")
print(report)

# คำนวณ AUC Score
y_true_onehot = to_categorical(y_true, num_classes=len(test_generator.class_indices))
auc_score = roc_auc_score(y_true_onehot, y_pred_prob, multi_class="ovr")
print(f"\nAUC Score: {auc_score:.4f}")
