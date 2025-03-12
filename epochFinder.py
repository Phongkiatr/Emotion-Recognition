import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ฟังก์ชันสร้างโมเดล CNN
def create_model():
    model = models.Sequential()
    model.add(layers.Input(shape=(48, 48, 1)))  # ขนาด input
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

# Data Augmentation
train_val_datagen = ImageDataGenerator(
    rescale=1./255, rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
    shear_range=0.1, zoom_range=0.1, horizontal_flip=True, fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)

# โหลดข้อมูล
train_generator = train_val_datagen.flow_from_directory(
    'Dataset_Split/train/', target_size=(48, 48), color_mode='grayscale', class_mode='sparse', batch_size=32
)
validation_generator = train_val_datagen.flow_from_directory(
    'Dataset_Split/val/', target_size=(48, 48), color_mode='grayscale', class_mode='sparse', batch_size=32
)
test_generator = test_datagen.flow_from_directory(
    'Dataset_Split/test/', target_size=(48, 48), color_mode='grayscale', class_mode='sparse', batch_size=32, shuffle=False
)

# ลองฝึกที่ epochs = 10, 20, 30
epochs_list = [10, 20, 30, 40, 50]
history_dict = {}

for epochs in epochs_list:
    print(f"Training with {epochs} epochs...")
    model = create_model()
    history = model.fit(
        train_generator, epochs=epochs, steps_per_epoch=len(train_generator),
        validation_data=validation_generator, validation_steps=len(validation_generator)
    )
    
    # เก็บค่าผลลัพธ์
    history_dict[epochs] = history.history

    # ทดสอบโมเดล
    test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_generator))
    print(f'Test Accuracy after {epochs} epochs: {test_accuracy:.4f}\n')

# พล็อตกราฟเปรียบเทียบ
plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
for epochs, hist in history_dict.items():
    plt.plot(hist['accuracy'], label=f'Train {epochs} epochs')
    plt.plot(hist['val_accuracy'], '--', label=f'Val {epochs} epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Comparison of Training & Validation Accuracy')
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
for epochs, hist in history_dict.items():
    plt.plot(hist['loss'], label=f'Train {epochs} epochs')
    plt.plot(hist['val_loss'], '--', label=f'Val {epochs} epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Comparison of Training & Validation Loss')
plt.legend()

plt.show()
