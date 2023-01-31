import os
import glob
import math
import random
import torch
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import cv2


def resize_square(img, height=416, color=(0, 0, 0)):  # resize a rectangular image to a padded square
	shape = img.shape[:2]  # shape = [height, width]
	ratio = float(height) / max(shape)  # ratio  = old / new
	new_shape = [round(shape[0] * ratio), round(shape[1] * ratio)]
	dw = height - new_shape[1]  # width padding
	dh = height - new_shape[0]  # height padding
	top, bottom = dh // 2, dh - (dh // 2)
	left, right = dw // 2, dw - (dw // 2)
	img = cv2.resize(img, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_AREA)  # resized, no border
	return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color), ratio, dw // 2, dh // 2


def random_affine(img, targets=None, degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-3, 3),
				  borderValue=(127.5, 127.5, 127.5)):
	# torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
	# https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
	border = 0  # width of added border (optional)
	height = max(img.shape[0], img.shape[1]) + border * 2

	# Rotation and Scale
	R = np.eye(3)
	a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
	# a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
	s = random.random() * (scale[1] - scale[0]) + scale[0]
	R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

	# Translation
	T = np.eye(3)
	T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border  # x translation (pixels)
	T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border  # y translation (pixels)

	# Shear
	S = np.eye(3)
	S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
	S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

	M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
	imw = cv2.warpPerspective(img, M, dsize=(height, height), flags=cv2.INTER_LINEAR,
							  borderValue=borderValue)  # BGR order borderValue

	# Return warped points also
	if targets is not None:
		if len(targets) > 0:
			n = targets.shape[0]
			points = targets[:, 1:9].copy()

			# warp points
			xy = np.ones((n * 4, 3))
			xy[:, :2] = points.reshape(n * 4, 2)

			xy = (xy @ M.T)[:, :2].reshape(n, 8)			

			# apply angle-based reduction
			radians = a * math.pi / 180
			reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
			
			x_center = np.mean(np.concatenate((np.max(xy[:, [0, 2, 4, 6]], 1).reshape(n, 1), np.min(xy[:, [0, 2, 4, 6]], 1).reshape(n, 1)), 1), 1)
			y_center = np.mean(np.concatenate((np.max(xy[:, [1, 3, 5, 7]], 1).reshape(n, 1), np.min(xy[:, [1, 3, 5, 7]], 1).reshape(n, 1)), 1), 1)		

			w_diff = (xy[:, [0, 2, 4, 6]] - x_center.reshape(n, 1)) * reduction
			h_diff = (xy[:, [1, 3, 5, 7]] - y_center.reshape(n, 1)) * reduction

			xy[:, [0, 2, 4, 6]] = x_center.reshape(n, 1) + w_diff
			xy[:, [1, 3, 5, 7]] = y_center.reshape(n, 1) + h_diff		

			# reject warped points outside of image
			np.clip(xy, 0, height, out=xy)
			i = []
			for k in range(0, xy.shape[0]):
				polygon1 = Polygon(xy[k,:].reshape(4,2)).convex_hull
				polygon2 = Polygon(points[k,:].reshape(4,2)).convex_hull
				i.append(polygon1.area / (polygon2.area + 1e-16) > 0.1)

			targets = targets[i]
			targets[:, 1:9] = xy[i]

		return imw, targets, M
	else:
		return imw



class load_images():
    def __init__(self, img_dir, batch_size=1, img_size=416):
        if os.path.isdir(img_dir):
            self.files = sorted(glob.glob('%s*.*'%img_dir))
        elif os.path.isfile(img_dir):
            self.files = [img_dir]
        self.nF = len(self.files)  # number of image files
        self.nB = math.ceil(self.nF / batch_size)  # number of batches
        self.batch_size = batch_size
        self.height = img_size
        assert self.nF > 0, 'No images found in path %s' % img_dir
    
    def __len__(self):
        return self.nB
    
    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count+=1
        if self.count == self.nB:
            raise StopIteration
        
        imgs_all = []
        paths_all = []
        ia = self.count * self.batch_size
        ib = min((self.count + 1) * self.batch_size, self.nF)
        for idx in range(ia, ib):
            img_path = self.files[idx]
            img = cv2.imread(img_path)  # BGR
            img, _, _, _ = resize_square(img, height=self.height, color=(127.5, 127.5, 127.5))  # padded resize
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
            img = np.ascontiguousarray(img, dtype=np.float32)
            img /= 255.0
            imgs_all.append(img)
            paths_all.append(img_path)
        return torch.from_numpy(imgs_all), paths_all 
        
        
class load_images_and_labels():
    def __init__(self, img_dir, csv_file, batch_size=1, img_size=416, augment=False, multi_scale=False, 
                aug_hsv=True, aug_affine=True, lr_flip=True, plot_flag=False):
        self.img_dir = img_dir
        self.data = pd.read_csv(csv_file)
        self.img_files = self.data['image_name'].tolist()
        self.labels_data = [[float(0)]+[float(x) for x in pts.strip('[]').split(',')]
            for pts in self.data['points'].tolist()]
        self.nF = len(self.img_files)  # number of image files
        self.nB = math.ceil(self.nF / batch_size)  # number of batches
        self.batch_size = batch_size
        self.height = img_size
        self.augment = augment
        self.multi_scale = multi_scale
        self.aug_hsv = aug_hsv
        self.aug_affine = aug_affine
        self.lr_flip = lr_flip
        self.plot_flag = plot_flag

    def __len__(self):
        return self.nB

    def __iter__(self):
        self.count = -1
        self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
        return self

    def __next__(self):
        self.count +=1
        if self.count == self.nB:
            raise StopIteration
        ia = self.count * self.batch_size
        ib = min((self.count + 1) * self.batch_size, self.nF)

		# Multi-Scale Training
        if self.augment and self.multi_scale:
            height = random.choice(range(10, 20)) * 32  # 320 - 608 pixels
        else:
            height = self.height
        
        img_all = []
        labels_all = []
        for index, files_index in enumerate(range(ia, ib)):
            img_name = self.img_files[self.shuffled_vector[files_index]]
            label_raw = self.labels_data[self.shuffled_vector[files_index]]
            
            img = cv2.imread(os.path.join(self.img_dir, img_name))
            if img is None:
                continue

            # SV augmentation by 50%
            if self.augment and self.aug_hsv:
                fraction = 0.50
                img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                S = img_hsv[:, :, 1].astype(np.float32)
                V = img_hsv[:, :, 2].astype(np.float32)

                a = (random.random() * 2 - 1) * fraction + 1
                S *= a
                if a > 1:
                    np.clip(S, a_min=0, a_max=255, out=S)

                a = (random.random() * 2 - 1) * fraction + 1
                V *= a
                if a > 1:
                    np.clip(V, a_min=0, a_max=255, out=V)

                img_hsv[:, :, 1] = S.astype(np.uint8)
                img_hsv[:, :, 2] = V.astype(np.uint8)
                cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

            h, w, _ = img.shape
            img, ratio, padw, padh = resize_square(img, height=height, color=(127.5, 127.5, 127.5))

            if len(label_raw)==9:
                label_np = np.array([label_raw], dtype=np.float32)
                labels = label_np.copy()
                for i in range(1, 8, 2):
                    labels[:, i] = ratio * label_np[:, i] + padw
                for j in range(2, 9, 2):
                    labels[:, j] = ratio * label_np[:, j] + padh
            else:
                labels = np.array([])
            
            # random affine image and labels
            aug_affine = True
            if self.augment and aug_affine:
                img, labels, M = random_affine(img, labels, degrees=(-5, 5), translate=(0.2, 0.2), scale=(0.8, 1.2))
            
            nL = len(labels)
            if nL > 0:
                labels[:, 1:9] = labels[:, 1:9] / height	

			# random left-right flip
            lr_flip = True
            if self.augment and lr_flip and (random.random() > 0.5):
                img = np.fliplr(img)
                if nL > 0:
                    labels[:, [1, 3, 5, 7]] = 1 - labels[:, [1, 3, 5, 7]]

            plotFlag = False
            if plotFlag:
                plt.figure(figsize=(10, 10))
                plt.imshow(img[:, :, ::-1])
                plt.plot(labels[:, [1, 3, 5, 7, 1]].T * height, labels[:, [2, 4, 6, 8, 2]].T * height, '.-')
                plt.axis('off')
                plt.show()

            img_all.append(img)
            labels_all.append(torch.from_numpy(labels))

        # Normalize
        img_all = np.stack(img_all)[:, :, :, ::-1].transpose(0, 3, 1, 2)
        img_all = np.ascontiguousarray(img_all, dtype=np.float32)
        img_all /= 255.0

        return torch.from_numpy(img_all), labels_all

