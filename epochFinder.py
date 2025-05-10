import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# สร้างโมเดล CNN
def create_model():
    model = models.Sequential()
    model.add(layers.Input(shape=(48, 48, 1)))  # กำหนดขนาดของ input
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))  # ใช้ 3 คลาส (อารมณ์)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Data Augmentation สำหรับ Train และ Validation
train_val_datagen = ImageDataGenerator(
    rescale=1./255,  # ปรับขนาดภาพ
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Data Augmentation สำหรับ Test (ไม่มีการ Augment)
test_datagen = ImageDataGenerator(rescale=1./255)

# โหลดข้อมูล Training (จาก Train Set)
train_generator = train_val_datagen.flow_from_directory(
    'Dataset_Split/train/',  # พาธไปยัง Train folder
    target_size=(48, 48),
    color_mode='grayscale',
    class_mode='sparse',
    batch_size=32
)

# โหลดข้อมูล Validation (จาก Validation Set)
validation_generator = train_val_datagen.flow_from_directory(
    'Dataset_Split/val/',  # พาธไปยัง Validation folder
    target_size=(48, 48),
    color_mode='grayscale',
    class_mode='sparse',
    batch_size=32
)

# โหลดข้อมูล Test (จาก Test Set)
test_generator = test_datagen.flow_from_directory(
    'Dataset_Split/test/',  # พาธไปยัง Test folder
    target_size=(48, 48),
    color_mode='grayscale',
    class_mode='sparse',
    batch_size=32,
    shuffle=False  # Test ไม่ต้องสลับรูป
)

# สร้างโมเดลและฝึกมัน
history = create_model().fit(
    train_generator,
    epochs=100,
    steps_per_epoch=len(train_generator),
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)

# ดึงข้อมูล accuracy และ loss จาก history
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# กำหนดค่า epoch ที่ต้องการพล็อต
epochs_to_plot = [10, 20, 30, 40, 50, 60]

# พล็อตกราฟ Accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_to_plot, [train_accuracy[e-1] for e in epochs_to_plot], label='Train Accuracy', marker='o')
plt.plot(epochs_to_plot, [val_accuracy[e-1] for e in epochs_to_plot], label='Validation Accuracy', marker='o')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# พล็อตกราฟ Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_to_plot, [train_loss[e-1] for e in epochs_to_plot], label='Train Loss', marker='o')
plt.plot(epochs_to_plot, [val_loss[e-1] for e in epochs_to_plot], label='Validation Loss', marker='o')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# บันทึกโมเดล
model.save('Model/model.h5')

# ทดสอบโมเดลบน Test Set
test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_generator))

# แสดงผลลัพธ์
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')
