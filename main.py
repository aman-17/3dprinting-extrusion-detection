import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

class ImageDataset:
    def __init__(self, csv_path, images_base_path):
        self.csv_path = csv_path
        self.images_base_path = images_base_path
        self.images = []
        self.labels = []
        
    def load_dataset(self):
        data = pd.read_csv(self.csv_path)
        for (i,j) in zip(data['img_path'], data['has_under_extrusion']):
            self.images.append(self.images_base_path + i)
            self.labels.append(j)
    
    def preprocess(self):
        for i in self.images:
            img = cv2.imread(i)
            img_array = cv2.resize(img, dsize=(224, 224)) 
            print(img_array)

        
        

def main():
    dataset = ImageDataset('./dataset/train.csv', './dataset/images/')
    dataset.load_dataset()
    dataset.preprocess()
    print(dataset.images)
    print(dataset.labels)

if __name__ == '__main__':
    main()
