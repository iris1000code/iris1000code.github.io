#8Unit08_3hand3.py
import cv2
import mediapipe as mp
import random
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 取得隨機粉嫩色系 (也包含更粉紅的範圍)
def get_pastel_pink():
    b = random.randint(180, 230)
    g = random.randint(150, 200)
    r = random.randint(220, 255)
    return (b, g, r)

# 繪製愛心圖案的函式
def draw_heart(img, center, size, color, thickness=-1):
    cx, cy = center
    # 愛心頂部的兩個圓形直徑
    r = size // 2
    # 定義愛心的多邊形頂點 (下方尖角處)
    pts = np.array([
        [cx - size, cy - r // 2],
        [cx + size, cy - r // 2],
        [cx, cy + size]
    ], np.int32)
    # 畫兩個圓
    cv2.circle(img, (cx - r, cy - r), r, color, thickness, cv2.LINE_AA)
    cv2.circle(img, (cx + r, cy - r), r, color, thickness, cv2.LINE_AA)
    # 畫下方的三角形
    cv2.fillPoly(img, [pts], color, cv2.LINE_AA)

# 初始化漂浮愛心資料
num_hearts = 25
floating_hearts = []
for _ in range(num_hearts):
    floating_hearts.append({
        'pos': [random.randint(0, 640), random.randint(0, 420)],
        'speed': random.uniform(1.0, 3.0),
        'size': random.randint(5, 12),
        'color': get_pastel_pink(),
        'alpha': random.uniform(0.3, 0.6)
    })

base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
landmarker = vision.HandLandmarker.create_from_options(options)
hand_connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

cap = cv2.VideoCapture(0)
run = True
rx, ry, count = 0, 0, 0
box_color = (180, 180, 255)
landmark_colors = [get_pastel_pink() for _ in range(21)]
connection_colors = [get_pastel_pink() for _ in range(len(hand_connections))]

while cap.isOpened():
    success, frame = cap.read()
    if not success: continue
    
    img = cv2.resize(frame, (640, 420))
    h, w, _ = img.shape
    
    # 1. 繪製並更新背景漂浮愛心
    overlay = img.copy()
    for h_obj in floating_hearts:
        # 繪製愛心
        draw_heart(overlay, tuple(map(int, h_obj['pos'])), h_obj['size'], h_obj['color'], -1)
        # 更新位置 (向上飄)
        h_obj['pos'][1] -= h_obj['speed']
        if h_obj['pos'][1] < -20: # 飄出畫面後重置到底部
            h_obj['pos'][1] = h + 20
            h_obj['pos'][0] = random.randint(0, w)
    # 融合背景
    img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

    if run:
        run = False
        rx = random.randint(10, w - 80)
        ry = random.randint(10, h - 80)
        box_color = get_pastel_pink()
    
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgrgb)
    detection_result = landmarker.detect(mp_image)
    
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            pixel_landmarks = []
            for lm in hand_landmarks:
                pixel_landmarks.append((int(lm.x * w), int(lm.y * h)))
            
            # 2. 骨架與節點繪製
            for idx, connection in enumerate(hand_connections):
                start_p = pixel_landmarks[connection.start]
                end_p = pixel_landmarks[connection.end]
                cv2.line(img, start_p, end_p, connection_colors[idx], 1, cv2.LINE_AA)
            
            for i, point in enumerate(pixel_landmarks):
                p_color = landmark_colors[i]
                if i == 20: # 小指尖：愛心雙圈效果
                    draw_heart(img, point, 10, (255, 255, 255), 1) # 白色愛心邊
                    cv2.circle(img, point, 4, p_color, -1, cv2.LINE_AA) # 中心小粉點
                elif i in [4, 8, 12, 16]: # 其他指尖：小愛心
                    draw_heart(img, point, 6, p_color, -1)
                else:
                    cv2.circle(img, point, 2, p_color, -1, cv2.LINE_AA)
            
            # 小指尖碰撞偵測
            if len(pixel_landmarks) >= 21:
                px, py = pixel_landmarks[20]
                if (rx < px < rx + 80) and (ry < py < ry + 80):
                    count += 1
                    run = True

    # 3. 目標框與分數顯示
    box_overlay = img.copy()
    cv2.rectangle(box_overlay, (rx, ry), (rx + 80, ry + 80), box_color, -1)
    img = cv2.addWeighted(box_overlay, 0.3, img, 0.7, 0)
    cv2.rectangle(img, (rx, ry), (rx + 80, ry + 80), (255, 255, 255), 1, cv2.LINE_AA)
    
    img = cv2.flip(img, 1)
    cv2.putText(img, f'Hearts: {count}', (25, 65), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1.5, (100, 100, 255), 2, cv2.LINE_AA)
    
    cv2.imshow('Unit08_3 | Heart Style | M11408041', img)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
landmarker.close()
cv2.destroyAllWindows()
