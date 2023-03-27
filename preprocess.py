import glob
import cv2
import os

#021
# path = "./dataset/images/021/1673025450/"
# for filename in glob.glob(path+'/*.jpg'):
# 	img = cv2.imread(filename)
# 	crop_img = img[120:1000, 220:1100]
# 	cv2.imwrite(path+filename,crop_img)
# 	# cv2.waitKey(0)

#022,023,102
path = "./dataset/images/102/1678736713/"
for filename in glob.glob(path+'/*.jpg'):
	img = cv2.imread(filename)
	crop_img = img[20:600, 220:1100]
	cv2.imshow("cropped", crop_img)
	# cv2.imwrite(path+filename,crop_img)
	cv2.waitKey(0)
