import cv2
import numpy as np
from mss import MSS
import pyautogui
import time
import threading

# Взятие размеров экрана
screen_width, screen_height = pyautogui.size()
monitor = {
    "top":    int(screen_width * 0.1),
    "left":   int(screen_height * 0.1),
    "width":  int(screen_width * 0.8),
    "height": int(screen_height * 0.8),
}

# Переменные
trex_bbox = None                        # Контур динозавра (прямоугольник)
frame_full = None                       # Бинарный кадр (0 - чёрный пиксель)
dino_w = 0                              # Ширина динозавра в пикселях (заполнится после клика)
field_w = 0                             # Ширина игрового поля в пикселях (заполнится после клика)

position_TRex = [0, 0]
clicked_TRex = False            
far_was_triggered = False               # Триггер дальней зоны
near_was_triggered = False              # Триггер ближней зоны

game_field_bbox = None                  # Координаты игрового поля (x1, y1, x2, y2)

# Зоны на игровом поле
JUMP_ZONE = (0.12, 0.70, 0.03, 0.2)     # Основная зона прыжка (x, y, w, h)
FAR_ZONE = (0.80, 0.70, 0.07, 0.2)      # Зона измерения скорости (начало)
NEAR_ZONE = (0.60, 0.70, 0.07, 0.2)     # Зона измерения скорости (конец)

EM_LEFT = 0.08                          # Левая граница аварийной зоны

# Таймер, скорость
timer_start = None          
last_elapsed = None         
known_distance_x = None                 # Расстояние между зонами FAR и NEAR

# Коэффициенты зоны прыжка
BASE_COEF = 0.2                         # Коэф прыжка (обычный)
EXTRA_COEF = 0.35                       # Коэф прыжка (при высокой скорости)
SPEED_THRESHOLD = 350                   # Нижний порог до увеличенного коэфф.

# Перезарядка прыжка
JUMP_COOLDOWN = 0.075                   # секунд между прыжками
last_jump_time = 0 

# Обработчик клика (выбор динозавра)
def on_click(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at {x}, {y}")
        global position_TRex, clicked_TRex
        position_TRex = [x, y]
        clicked_TRex = True

# Окна OpenCV
window_width, window_height = 640, 360
cv2.namedWindow("T-Rex Game", cv2.WINDOW_NORMAL)
cv2.resizeWindow("T-Rex Game", window_width, window_height)
cv2.setMouseCallback("T-Rex Game", on_click)

print("Кликните на динозавра для выделения")
print("ESC - выход")


# ГЛАВНЫЙ ЦИКЛ

with MSS() as sct:
    while True:
        # Захват и бинаризация
        img = sct.grab(monitor)
        frame_original = np.array(img)
        frame_full = cv2.cvtColor(frame_original, cv2.COLOR_BGRA2GRAY)
        _, frame_full = cv2.threshold(frame_full, 127, 255, cv2.THRESH_BINARY)

        # Обработка клика (выделение динозавра)
        if clicked_TRex:
            x, y = position_TRex
            h, w = frame_full.shape
            if 0 <= x < w and 0 <= y < h and frame_full[y, x] == 0:
                # Заливка области динозавра (для выделения)
                mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
                cv2.floodFill(frame_full, mask, (x, y), 255, loDiff=0, upDiff=0, flags=8)
                mask = mask[1:-1, 1:-1]
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    trex_bbox = cv2.boundingRect(contours[0])
                    print(f"Найден объект, bbox: {trex_bbox}")

                    x_b, y_b, w_b, h_b = trex_bbox
                    H, W = frame_original.shape[:2]

                    # Размеры игрового поля (15 динозавров в ширину и 3 в высоту)
                    field_w = w_b * 15
                    field_h = h_b * 3

                    dino_left = x_b
                    dino_bottom = y_b + h_b

                    x1 = dino_left
                    y2 = dino_bottom
                    x2 = x1 + field_w
                    y1 = y2 - field_h

                    # Коррекция границ поля
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(W, x2)
                    y2 = min(H, y2)

                    game_field_bbox = (x1, y1, x2, y2)

                    # Расстояние между центрами измерительных зон
                    fw = x2 - x1
                    far_cx = x1 + int(FAR_ZONE[0] * fw) + int(FAR_ZONE[2] * fw) // 2
                    near_cx = x1 + int(NEAR_ZONE[0] * fw) + int(NEAR_ZONE[2] * fw) // 2
                    known_distance_x = abs(far_cx - near_cx)
                    print(f"Distance between zones: {known_distance_x} px")

                    dino_w = w_b
                    field_w = fw
                else:
                    trex_bbox = None
            else:
                print("Пиксель не чёрный или вне кадра")
            clicked_TRex = False

        # Основное окно с рамкой динозавра
        if trex_bbox is not None:
            x_b, y_b, w_b, h_b = trex_bbox
            disp_frame = cv2.cvtColor(frame_full, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(disp_frame, (x_b, y_b), (x_b + w_b, y_b + h_b), (0, 255, 0), 2)
            cv2.imshow("T-Rex Game", disp_frame)
        else:
            cv2.imshow("T-Rex Game", frame_full)

        # Обработка игрового поля
        if game_field_bbox is not None:
            x1, y1, x2, y2 = game_field_bbox
            game_field = frame_original[y1:y2, x1:x2].copy()
            fw = x2 - x1
            fh = y2 - y1
            H, W = frame_full.shape

            # Основная зона прыжка
            jx = x1 + int(JUMP_ZONE[0] * fw)
            jy = y1 + int(JUMP_ZONE[1] * fh)
            jw = int(JUMP_ZONE[2] * fw)
            jh = int(JUMP_ZONE[3] * fh)
            jx2 = min(jx + jw, W)
            jy2 = min(jy + jh, H)

            # Дальняя зона
            far_x = x1 + int(FAR_ZONE[0] * fw)
            far_y = y1 + int(FAR_ZONE[1] * fh)
            far_w = int(FAR_ZONE[2] * fw)
            far_h = int(FAR_ZONE[3] * fh)
            far_x2 = min(far_x + far_w, W)
            far_y2 = min(far_y + far_h, H)

            # Ближняя зона
            near_x = x1 + int(NEAR_ZONE[0] * fw)
            near_y = y1 + int(NEAR_ZONE[1] * fh)
            near_w = int(NEAR_ZONE[2] * fw)
            near_h = int(NEAR_ZONE[3] * fh)
            near_x2 = min(near_x + near_w, W)
            near_y2 = min(near_y + near_h, H)

            # Аварийная зона
            em_x = x1 + int(EM_LEFT * fw)
            em_y = jy
            em_h = jh
            em_x2 = jx
            if em_x2 <= em_x:
                em_x2 = em_x + 1

            # Отрисовка всех зон
            cv2.rectangle(game_field, (jx - x1, jy - y1), (jx - x1 + jw, jy - y1 + jh), (0, 0, 255), 2)             # красный - основная
            cv2.rectangle(game_field, (far_x - x1, far_y - y1), (far_x2 - x1, far_y2 - y1), (255, 0, 0), 2)         # синий - дальняя
            cv2.rectangle(game_field, (near_x - x1, near_y - y1), (near_x2 - x1, near_y2 - y1), (0, 255, 0), 2)     # зелёный - ближняя
            cv2.rectangle(game_field, (em_x - x1, em_y - y1), (em_x2 - x1, em_y + em_h - y1), (0, 255, 255), 2)     # жёлтый - аварийная

            # Таймер (измерение скорости)
            far_black = np.any(frame_full[far_y:far_y2, far_x:far_x2] == 0)
            near_black = np.any(frame_full[near_y:near_y2, near_x:near_x2] == 0)

            # Передний край дальней зоны (появление препятствия)
            if far_black and not far_was_triggered and timer_start is None:
                timer_start = time.time()
            far_was_triggered = far_black

            # Задний край ближней зоны (исчезновение препятствия)
            if not near_black and near_was_triggered and timer_start is not None:
                last_elapsed = time.time() - timer_start
                timer_start = None
                print(f"Time: {last_elapsed:.3f}s")

                # Адаптивное смещение основной зоны
                if last_elapsed > 0 and known_distance_x is not None:
                    v = known_distance_x / last_elapsed
                    if v <= SPEED_THRESHOLD:
                        dist_px = v * BASE_COEF
                    else:
                        dist_px = v * BASE_COEF + (v - SPEED_THRESHOLD) * EXTRA_COEF

                    new_zone_x = (dino_w + dist_px) / field_w
                    new_zone_x = min(0.95, new_zone_x)
                    new_zone_x = max(0.05, new_zone_x)
                    JUMP_ZONE = (new_zone_x, JUMP_ZONE[1], JUMP_ZONE[2], JUMP_ZONE[3])

            near_was_triggered = near_black

            # Прыжок
            current_time = time.time()
            main_trigger = np.any(frame_full[jy:jy2, jx:jx2] == 0)
            emerg_trigger = np.any(frame_full[em_y:em_y+em_h, em_x:em_x2] == 0)

            if (main_trigger or emerg_trigger) and (current_time - last_jump_time > JUMP_COOLDOWN):
                threading.Thread(target=pyautogui.press, args=('up',)).start()
                last_jump_time = current_time

            cv2.imshow("Game Field", game_field)

        # Выход по ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

cv2.destroyAllWindows()