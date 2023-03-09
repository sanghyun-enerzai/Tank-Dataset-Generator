# -*- coding: utf-8 -*-
import os
import gc
import cv2
import math
import random
import trimesh
import numpy as np

from pathlib import Path

os.environ['PYOPENGL_PLATFORM'] = 'egl'


class OBJ2PNG(object):
    def __init__(self, obj_file: str, quality: str):
        assert quality in ['h', 'm', 'l']
        self.obj_file = Path(obj_file)
        self.mesh = trimesh.load(self.obj_file)
        self.scene = trimesh.Scene(self.mesh)
        self.corners = self.scene.bounds_corners[self.obj_file.name]
        self.quality = quality
        self.centroid = self.mesh.centroid

        self.resolution = {
            'h': (1200, 1200),
            'm': (700, 700),
            'l': (300, 300),
        }[self.quality]

    def get_png_image(self, size=None):
        x_rand = math.radians(random.random() * 360)
        y_rand = math.radians((1 - (1 - random.random()) ** 1.3) * -90)
        r_e = trimesh.transformations.euler_matrix(x_rand, y_rand, 0, 'ryxz')
        d_e = trimesh.transformations.scale_matrix(factor=1.3)
        self.scene.set_camera(center=self.centroid)
        t_r = self.scene.camera.look_at(self.corners, rotation=np.dot(r_e, d_e))
        self.scene.camera_transform = t_r

        png = self.scene.save_image(resolution=self.resolution, visible=True)
        png = np.frombuffer(png, dtype=np.uint8)
        img = cv2.imdecode(png, cv2.IMREAD_COLOR)
        tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(tmp, 254, 255, cv2.THRESH_BINARY_INV)
        x, y, w, h = cv2.boundingRect(alpha)
        alpha = alpha[y:y + h, x:x + w]
        img = img[y:y + h, x:x + w, :]

        b, g, r = cv2.split(img)
        rgba = [b, g, r, alpha]
        dst = cv2.merge(rgba, 4)
        if size is not None:
            scale_factor = size / min(w, h)
            reshape_size = (int(w * scale_factor) - 2, int(h * scale_factor) - 2)
            dst = cv2.resize(dst, reshape_size, interpolation=cv2.INTER_AREA)
        return dst

    @staticmethod
    def gamma_correction(self, img: any, gamma: float):
        inv_gamma = 1 / gamma
        table = [(i / 255) ** inv_gamma * 255 for i in range(256)]
        table = np.array(table, dtype=np.uint8)
        return cv2.LUT(img, table)
