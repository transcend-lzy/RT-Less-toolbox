import cv2
import  numpy as np


data_path="/home/reflex/PycharmProjects/KeypointDetection/640*480/test/obj1/photo_cut/16.png"

img=cv2.imread(data_path)

new_img=img[150:260,220:280]


cv2.imshow("test",new_img)
B=0
G=0
R=0
count =0
for i in range(150,260):
    for j in range(220,280):
        B=B+img[i][j][0]
        G=G+img[i][j][1]
        R=R+img[i][j][2]
        count=count+1
B=B/count
G = G / count
R = R / count

print(B)
print(G)
print(R)
cv2.waitKey()