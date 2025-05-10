from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# สร้างโมเดล CNN
def create_model():
    model = models.Sequential()
    model.add(layers.Input(shape=(48, 48, 1)))  # ขนาดของ input
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

# สร้างและฝึกโมเดล
model = create_model()
model.fit(
    train_generator,
    epochs=50,
    steps_per_epoch=len(train_generator),
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)

# บันทึกโมเดล
model.save('Model/model.h5')

# ทดสอบโมเดลบน Test Set
test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_generator))

# แสดงผลลัพธ์
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')
