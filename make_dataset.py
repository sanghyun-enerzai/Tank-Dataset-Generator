# -*- coding: utf-8 -*-
import cv2
import json
import random

from tqdm import tqdm
from glob import glob
from pathlib import Path
from scripts.obj2png import OBJ2PNG
from scripts.get_scene import get_scene

if __name__ == '__main__':
    quality = 'm'
    version = 'v1.0'

    quality_long = {'l': 'Low', 'm': 'Middle', 'h': 'High'}[quality]
    dataset_path = Path(f'tank_coco_{quality}_{version}')
    dataset_path.mkdir(parents=True, exist_ok=True)
    img_path = dataset_path / 'data'
    img_path.mkdir(parents=True, exist_ok=True)
    num_pictures = int(input('How many pictures?: '))
    tank0_maker = OBJ2PNG('objects/tank0/textured.obj', quality=quality)
    tank1_maker = OBJ2PNG('objects/tank1/textured.obj', quality=quality)
    tank2_maker = OBJ2PNG('objects/tank2/textured.obj', quality=quality)
    tank3_maker = OBJ2PNG('objects/tank3/textured.obj', quality=quality)
    backgrounds = glob('backgrounds/test2017/*.jpg')

    categories = [{
        'supercategory': 'Tank',
        'id': i,
        'name': f'Tank{i}',
    } for i in range(4)]

    categories2 = [{
        'supercategory': 'Enemy',
        'id': 0,
        'name': f'Enemy',
    }]

    info = {
        'year': '2023',
        'version': version,
        'description': f'Tank Dataset {quality_long} {version}',
        'date_created': '2023-02-08',
    }

    train_dset = dict()
    train_dset['info'] = info
    train_dset['images'] = []
    train_dset['categories'] = categories
    train_dset['annotations'] = []

    train_onelabel_dset = dict()
    train_onelabel_dset['info'] = info
    train_onelabel_dset['images'] = []
    train_onelabel_dset['categories'] = categories2
    train_onelabel_dset['annotations'] = []

    val_dset = dict()
    val_dset['info'] = info
    val_dset['images'] = []
    val_dset['categories'] = categories
    val_dset['annotations'] = []

    val_onelabel_dset = dict()
    val_onelabel_dset['info'] = info
    val_onelabel_dset['images'] = []
    val_onelabel_dset['categories'] = categories2
    val_onelabel_dset['annotations'] = []

    anno_id = 0

    for i in tqdm(range(num_pictures)):
        data_type = 'val' if random.random() < 0.1 else 'train'
        bg_path = random.choice(backgrounds)
        bg_img = cv2.imread(bg_path)
        scene, annotation = get_scene(bg_img, [tank0_maker, tank1_maker, tank2_maker, tank3_maker])
        file_name = f'scene_{i}.jpg'
        cv2.imwrite(str(img_path / file_name), scene)
        h, w, c = scene.shape
        new_data_img = {
            'file_name': file_name,
            'height': h,
            'width': w,
            'id': i
        }
        assert len(annotation) == 4
        if data_type == 'val':
            val_dset['images'].append(new_data_img)
            val_onelabel_dset['images'].append(new_data_img)
            for j in range(4):
                val_dset['annotations'].append({
                    'id': anno_id,
                    'image_id': i,
                    'bbox': annotation[j],
                    'area': annotation[j][2] * annotation[j][3],
                    'iscrowd': 0,
                    'category_id': j,
                    'segmentation': [],
                })
                val_onelabel_dset['annotations'].append({
                    'id': anno_id,
                    'image_id': i,
                    'bbox': annotation[j],
                    'area': annotation[j][2] * annotation[j][3],
                    'iscrowd': 0,
                    'category_id': 0,
                    'segmentation': [],
                })

                anno_id += 1
        else:
            train_dset['images'].append(new_data_img)
            train_onelabel_dset['images'].append(new_data_img)
            for j in range(4):
                train_dset['annotations'].append({
                    'id': anno_id,
                    'image_id': i,
                    'bbox': annotation[j],
                    'area': annotation[j][2] * annotation[j][3],
                    'iscrowd': 0,
                    'category_id': j,
                    'segmentation': [],
                })
                train_onelabel_dset['annotations'].append({
                    'id': anno_id,
                    'image_id': i,
                    'bbox': annotation[j],
                    'area': annotation[j][2] * annotation[j][3],
                    'iscrowd': 0,
                    'category_id': 0,
                    'segmentation': [],
                })
                anno_id += 1

    with open(dataset_path / 'train.json', 'w') as f:
        json.dump(train_dset, f, indent=4)

    with open(dataset_path / 'val.json', 'w') as f:
        json.dump(val_dset, f, indent=4)

    with open(dataset_path / 'train_onelabel.json', 'w') as f:
        json.dump(train_onelabel_dset, f, indent=4)

    with open(dataset_path / 'val_onelabel.json', 'w') as f:
        json.dump(val_onelabel_dset, f, indent=4)


