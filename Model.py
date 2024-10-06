from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# สร้างโมเดล CNN
def create_model():
    model = models.Sequential()
    model.add(layers.Input(shape=(48, 48, 1)))  # เปลี่ยนจาก input_shape มาใช้ Input แทน
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(7, activation='softmax'))  # เปลี่ยนจำนวนออกเป็น 7 สำหรับ 7 อารมณ์

    # คอมไพล์โมเดล
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# สร้าง ImageDataGenerator สำหรับการโหลดและทำ Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalization
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# โหลดข้อมูลการฝึก
train_generator = train_datagen.flow_from_directory(
    'DataSet/CKPlusDataset/',  # พาธที่เก็บโฟลเดอร์ของข้อมูล
    target_size=(48, 48),
    color_mode='grayscale',
    class_mode='sparse',
    batch_size=32
)

# สร้างและฝึกโมเดล
model = create_model()
model.fit(train_generator, epochs=20, steps_per_epoch=len(train_generator))

# บันทึกโมเดล
model.save('Model/CK_model.h5')