import cv2
import numpy as np

#Carrega a imagem
img = cv2.imread('Defina aqui o diretorio para carregar a imagem')
img_original = img.copy()
GrayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #Transforma para tons de Cinza
BlurredFrame = cv2.GaussianBlur(GrayImg, (5, 5), 1) #aplica um desfoque gaussiano na imagem
_, binary = cv2.threshold(BlurredFrame, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Determina os pontos dos contornos de interesse
ContourFrame = img.copy()
ContourFrame = cv2.drawContours(ContourFrame, contours[0], -1, (255, 0, 0), 25) #Desenha o contorno de interesse
i = contours[0]
peri = cv2.arcLength(i, True)
edges = cv2.approxPolyDP(i, 0.02*peri, True)
CornerFrame = img.copy()
maxArea = 0
biggest = []
for i in contours :
    area = cv2.contourArea(i)
    if area > 1000 :
        peri = cv2.arcLength(i, True)
        edges = cv2.approxPolyDP(i, 0.02*peri, True)
        if area > maxArea:
            biggest = edges
            maxArea = area
if len(biggest) != 0 :
    CornerFrame = cv2.drawContours(CornerFrame, biggest, -1, (255, 0, 255), 25)
# cv2.drawContours(img, [biggest], -1, (0, 255, 0), 3)
# Pixel values in the original image
points = biggest.reshape(4, 2)
input_points = np.zeros((4, 2), dtype="float32")

points_sum = points.sum(axis=1)
input_points[0] = points[np.argmin(points_sum)]
input_points[3] = points[np.argmax(points_sum)]

points_diff = np.diff(points, axis=1)
input_points[1] = points[np.argmin(points_diff)]
input_points[2] = points[np.argmax(points_diff)]

(top_left, top_right, bottom_right, bottom_left) = input_points
bottom_width = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
top_width = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
right_height = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))
left_height = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))

# Output image size
max_width = max(int(bottom_width), int(top_width))
# max_height = max(int(right_height), int(left_height))
max_height = int(max_width * 1.414)  # for A4

# Desired points values in the output image
converted_points = np.float32([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]])

# Perspective transformation
matrix = cv2.getPerspectiveTransform(input_points, converted_points)
#print(matrix)
img_warp = cv2.warpPerspective(img_original, matrix, (max_width, max_height)) #Corta a imagem de acordo com o contorno determinado

cv2.imwrite('Defina aqui o diretório para salvar a imagem cortada', img_warp) 
