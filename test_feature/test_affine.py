

import cv2
import numpy as np
img = cv2.imread("D:/work/face_expression/test_feature/affine.jpg")
print img.shape
rows,cols,channel = img.shape
#rotate
M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
dst = cv2.warpAffine(img,M,(cols,rows))

#affine
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])
M = cv2.getAffineTransform(pts1,pts2)

dst = cv2.warpAffine(img,M,(cols,rows))



#perspective
rows,cols,ch = img.shape
pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,200],[300,300]])
M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(img,M,(300,300))
cv2.namedWindow("Image")
cv2.imshow('Image',dst)
cv2.waitKey (0)  
cv2.destroyAllWindows()  
