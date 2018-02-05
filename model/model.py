# code based on:
#YAD2K https://github.com/allanzelener/YAD2K
#darkflow https://github.com/thtrieu/darkflow
#Darknet.keras https://github.com/sunshineatnoon/Darknet.keras
#Real time vehicle detection using YOLO https://github.com/xslittlegrass/CarND-Vehicle-Detection

import numpy as np
import cv2
import keras # broken for keras >= 2.0, use 1.2
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Flatten, Dense, Activation, Reshape


class Box:
    def __init__(self):
        self.x, self.y = float(), float()
        self.w, self.h = float(), float()
        self.c = float()
        self.prob = float()

class TinyYOLO:
    def __init__(self, yolo_weight_file_path):
        self.yolo_weight_file = yolo_weight_file_path
        self.createModel()
        self.load_weights()

    def createModel(self):
        model = Sequential()
        model.add(Convolution2D(16, 3, 3, input_shape=(3, 448, 448), border_mode='same', subsample=(1, 1)))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(32, 3, 3, border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
        model.add(Convolution2D(64, 3, 3, border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
        model.add(Convolution2D(128, 3, 3, border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
        model.add(Convolution2D(256, 3, 3, border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
        model.add(Convolution2D(512, 3, 3, border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
        model.add(Convolution2D(1024, 3, 3, border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Convolution2D(1024, 3, 3, border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Convolution2D(1024, 3, 3, border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Dense(4096))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dense(1470))
        self.model = model

    def load_weights(self):
        data = np.fromfile(self.yolo_weight_file, np.float32)
        data = data[4:]

        index = 0
        for layer in self.model.layers:
            shape = [w.shape for w in layer.get_weights()]
            if shape != []:
                kshape, bshape = shape
                bia = data[index:index + np.prod(bshape)].reshape(bshape)
                index += np.prod(bshape)
                ker = data[index:index + np.prod(kshape)].reshape(kshape)
                index += np.prod(kshape)
                layer.set_weights([ker, bia])

    def box_union(self, a, b):
        i = self.box_intersection(a, b)
        u = a.w * a.h + b.w * b.h - i
        return u

    def box_intersection(self, a, b):
        w = self.overlap(a.x, a.w, b.x, b.w)
        h = self.overlap(a.y, a.h, b.y, b.h)
        if w < 0 or h < 0: return 0;
        area = w * h
        return area

    def overlap(self, x1, w1, x2, w2):
        l1 = x1 - w1 / 2.
        l2 = x2 - w2 / 2.
        left = max(l1, l2)
        r1 = x1 + w1 / 2.
        r2 = x2 + w2 / 2.
        right = min(r1, r2)
        return right - left


    def box_iou(self, a, b):
        return self.box_intersection(a, b) / self.box_union(a, b)


    def draw_box(self, boxes, im, crop_dim):
        imgcv = im
        [xmin, xmax] = crop_dim[0]
        [ymin, ymax] = crop_dim[1]
        for b in boxes:
            h, w, _ = imgcv.shape
            left = int((b.x - b.w / 2.) * w)
            right = int((b.x + b.w / 2.) * w)
            top = int((b.y - b.h / 2.) * h)
            bot = int((b.y + b.h / 2.) * h)
            left = int(left * (xmax - xmin) / w + xmin)
            right = int(right * (xmax - xmin) / w + xmin)
            top = int(top * (ymax - ymin) / h + ymin)
            bot = int(bot * (ymax - ymin) / h + ymin)

            if left < 0:  left = 0
            if right > w - 1: right = w - 1
            if top < 0:   top = 0
            if bot > h - 1:   bot = h - 1
            thick = int((h + w) // 150)
            cv2.rectangle(imgcv, (left, top), (right, bot), (255, 0, 0), thick)

        return imgcv

    def get_detected_boxes(self, net_out, threshold=0.2, sqrt=1.8, C=20, B=2, S=7):
        class_num = 6
        boxes = []
        SS = S * S  # number of grid cells
        prob_size = SS * C  # class probabilities
        conf_size = SS * B  # confidences for each grid cell

        probs = net_out[0: prob_size]
        confs = net_out[prob_size: (prob_size + conf_size)]
        cords = net_out[(prob_size + conf_size):]
        probs = probs.reshape([SS, C])
        confs = confs.reshape([SS, B])
        cords = cords.reshape([SS, B, 4])

        for grid in range(SS):
            for b in range(B):
                bx = Box()
                bx.c = confs[grid, b]
                bx.x = (cords[grid, b, 0] + grid % S) / S
                bx.y = (cords[grid, b, 1] + grid // S) / S
                bx.w = cords[grid, b, 2] ** sqrt
                bx.h = cords[grid, b, 3] ** sqrt
                p = probs[grid, :] * bx.c

                if p[class_num] >= threshold:
                    bx.prob = p[class_num]
                    boxes.append(bx)

        # combine boxes that are overlap
        boxes.sort(key=lambda b: b.prob, reverse=True)
        for i in range(len(boxes)):
            boxi = boxes[i]
            if boxi.prob == 0: continue
            for j in range(i + 1, len(boxes)):
                boxj = boxes[j]
                if self.box_iou(boxi, boxj) >= .4:
                    boxes[j].prob = 0.
        boxes = [b for b in boxes if b.prob > 0.]

        return boxes

