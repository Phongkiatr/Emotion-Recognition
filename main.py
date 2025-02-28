import cv2
import threading
from capture_and_display import capture_and_predict

# เปิดกล้อง
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 120)

# สร้างเทรดแยกสำหรับการจับภาพและทำนาย
capture_thread = threading.Thread(target=capture_and_predict, args=(cap,))
capture_thread.start()

capture_thread.join()
cap.release()
cv2.destroyAllWindows()
