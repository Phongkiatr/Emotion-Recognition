from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# โหลดโมเดลที่ฝึกเสร็จแล้ว
model = load_model('Model/model.h5')

# เตรียมข้อมูลสำหรับการทดสอบ (ไม่มีการ Augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    'Dataset_Split/test/',  # พาธไปยังโฟลเดอร์ Test
    target_size=(48, 48),
    color_mode='grayscale',
    class_mode='sparse',
    batch_size=32,
    shuffle=False  # Test ไม่ต้องสลับรูป
)

# ประเมินโมเดลที่โหลดมา
test_loss, test_accuracy = model.evaluate(test_generator)

# แสดงผลลัพธ์
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# ทำนายผลลัพธ์
y_pred = model.predict(test_generator, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)  # เปลี่ยนเป็นคลาสที่ทำนาย

# ค่าจริง (True labels)
y_true = test_generator.classes

# คำนวณ Precision, Recall, F1-score
report = classification_report(y_true, y_pred_classes, target_names=test_generator.class_indices.keys())
print(report)

from sklearn.metrics import roc_auc_score
from tensorflow.keras.utils import to_categorical

# แปลงค่า y_true เป็น one-hot encoding
y_true_onehot = to_categorical(y_true, num_classes=3)

# คำนวณ AUC Score
auc = roc_auc_score(y_true_onehot, model.predict(test_generator), multi_class='ovr')
print(f"AUC Score: {auc:.4f}")
