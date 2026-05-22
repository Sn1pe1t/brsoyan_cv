import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops
from skimage.io import imread
from pathlib import Path

def sanitize_folder_name(symbol):
    mapping = {
        '*': 'star',
        '/': 'slash',
        '-': 'minus',
        '?': 'unknown'
    }
    return mapping.get(symbol, symbol)

def count_holes(region):
    shape = region.image.shape
    new_image = np.zeros((shape[0] + 2, shape[1] + 2))
    new_image[1:-1, 1:-1] = region.image
    new_image = np.logical_not(new_image)
    labeled = label(new_image)
    return np.max(labeled) - 1

def classificator(region):
    holes = count_holes(region)
    h, w = region.image.shape
    aspect = w / h
    first_col = region.image[:, 0]
    fill_left = np.sum(first_col) / h
    bottom_row = region.image[-1, :]
    fill_bottom = np.sum(bottom_row) / w

    if holes == 2:
        # Различение "B" и "8" по левой границе
        if fill_left > 0.9:
            return "B"
        else:
            return "8"
    elif holes == 1:
        # Различение "A", "D", "P" и "0" по нижней и левой границам
        left = w // 3
        right = 2 * w // 3
        bottom_center = region.image[-1, left:right]
        center_fill = np.sum(bottom_center) / len(bottom_center)
        if fill_left < 0.5 and center_fill < 0.5:
            return "A"
        elif fill_left > 0.9 and fill_bottom > 0.5:
            return "D"
        elif fill_left > 0.9 and fill_bottom < 0.3:
            return "P"
        elif center_fill > 0.5:
            return "0"
    else:
        # Проверка на "1" по нижней границе
        if fill_bottom > 0.9 and aspect < 1.3:
            return "1"
        
        # Проверки на "-", "*" и "W" по соотношению сторон
        if aspect > 2:
            return "-"
        if 0.9 < aspect < 1.2:
            return "*"
        if 1.2 < aspect < 2:
            return "W"

        # Различение "X" и "/" по верхней и нижней границам
        top_row = region.image[0, :]
        fill_top = np.sum(top_row) / w
        if fill_top + fill_bottom > 0.9:
            return "X"
        else:
            return "/"
    
    return "?"

save_path = Path(__file__).parent

image = imread("symbols.png")[:, :, :-1]
abinary = image.mean(2) > 0
alabeled = label(abinary)
print("Всего символов:", np.max(alabeled))
aprops = regionprops(alabeled)

result = {}
image_path = save_path / "out"
image_path.mkdir(exist_ok=True)

# Очистка файлов .png во всех подпапках out
for class_dir in image_path.iterdir():
    if class_dir.is_dir():
        for file in class_dir.glob("*.png"):
            file.unlink()

plt.figure(figsize=(5, 7))
for region in aprops:
    symbol = classificator(region)
    if symbol not in result:
        result[symbol] = 0
    result[symbol] += 1

    plt.cla()
    plt.title(f"Class - '{symbol}'")
    plt.imshow(region.image)

    class_dir = image_path / sanitize_folder_name(symbol)
    class_dir.mkdir(exist_ok=True)
    plt.savefig(class_dir / f"image_{region.label}.png")

sorted_result_dict = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

print("Распознанные символы:\n", result)
print("Частотный словарь:\n", sorted_result_dict)

print(f"Процент распознанных символов: {int((1.0 - result.get('?', 0) / len(aprops)) * 100)}%")