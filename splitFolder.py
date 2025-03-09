import splitfolders

# แบ่งข้อมูล 70% Train, 15% Validation, 15% Test
splitfolders.ratio(
    "Dataset/",  # พาธที่เก็บข้อมูลต้นฉบับ
    output="Dataset_Split/",  # โฟลเดอร์ที่เก็บข้อมูลที่แบ่งแล้ว
    seed=42,
    ratio=(0.7, 0.15, 0.15)  # 70% Train, 15% Validation, 15% Test
)
