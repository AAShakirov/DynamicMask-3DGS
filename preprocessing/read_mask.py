import numpy as np

# Загрузка 5-канального изображения
image_4ch = np.load('image_4_ch.npy')

# Извлечение каналов
bgr_image = image_4ch[:, :, :3]  # BGR изображение
object_mask = image_4ch[:, :, 3]  # Бинарная маска объектов

x, y = 320, 240
if object_mask[y, x] > 0:
    print(f"({x}, {y}) - object")
else:
    print(f"({x}, {y}) - empty")

