# -*- coding: utf-8 -*-
import random
import numpy as np


def get_scene(background: np.array, object_makers: any):
    bg_h, bg_w, bg_c = background.shape
    positions = []
    for i in range(len(object_makers)):
        scale = random.random() * 0.075 + 0.065
        obj_png = object_makers[i].get_png_image(size=int(scale * min(bg_h, bg_w)))
        h, w, c = obj_png.shape
        x = random.randrange(0, bg_w - w)
        y = random.randrange(0, bg_h - h)

        while True:
            is_valid = True
            for pos in positions:
                box_x, box_y, box_w, box_h = pos
                x_over = (box_x <= x <= box_x + box_w) or (box_x <= x + w <= box_x + box_w)
                y_over = (box_y <= y <= box_y + box_h) or (box_y <= y + h <= box_y + box_h)
                box_x_over = (x <= box_x <= x + w) or (x <= box_x + box_w <= x + w)
                box_y_over = (y <= box_y <= y + h) or (y <= box_y + box_h <= y + h)
                if (x_over and y_over) or (box_x_over and box_y_over):
                    is_valid = False
                    break
            if is_valid:
                break
            else:
                x = random.randrange(0, bg_w - w)
                y = random.randrange(0, bg_h - h)

        opacity = obj_png[:, :, 3] / 255
        l = np.mean(obj_png[:, :, :3], axis=2)
        opacity[opacity < 0.1] = 0.0
        opacity[l > 240] = 0.0
        opacity **= 3
        for j in range(3):
            background[y:y+h, x:x+w, j] = obj_png[:, :, j] * opacity + background[y:y+h, x:x+w, j] * (1 - opacity)

        positions.append([x, y, w, h])

    return background, positions
