"""
Ordered Dithering and grayScale
-----------
:copyright: 2020-08-09 by MohammadHossein Nejadhendi <moh.nezh1377@gmail.com>
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import math


# calculate log2 of number
def Log2(x):
    return (math.log10(x) / 
            math.log10(2))
  
# Function to check
# if x is power of 2
def isPowerOfTwo(n):
    return (math.ceil(Log2(n)) == math.floor(Log2(n)))

# gray scale converter function_type 1
def GrayScaleConverter(image):
    # from Wikipedia _ Link : https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale
    return np.uint8(0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2])

# gray scale converter function_type 2
def GrayScaleConverter2(image):
    weights = [0.11, 0.59, 0.3]
    return np.uint8(np.dot(image, weights))

# generate dither matrix function
def GenerateDitherMatrix(n):
    if n == 1:
        return 0

    prevStateArr = np.array(GenerateDitherMatrix(n/2))
    # print('n is : '+str(n))

    # print(prevStateArr)

    sec1 = ((n ** 2)*prevStateArr)
    # print(sec1)

    sec2 = 2 + ((n ** 2)*prevStateArr)
    # print(sec2)

    row1 = np.array([sec1, sec2])
    if np.count_nonzero(sec1) > 1 and np.count_nonzero(sec2) > 1:
        row1 = np.hstack((sec1, sec2))

    sec3 = 3 + ((n ** 2)*prevStateArr)
    # print(sec3)

    sec4 = 1 + ((n ** 2)*prevStateArr)
    # print(sec4)

    row2 = np.array([sec3, sec4])
    if np.count_nonzero(sec3) > 1 and np.count_nonzero(sec4) > 1:
        row2 = np.hstack((sec3, sec4))

    return np.vstack((row1, row2)) * 1/(n ** 2)

# dithering with ordered alorthm
def OrderedDithering(arr,n):
    max_x = arr.shape[0]
    max_y = arr.shape[1]
    finalArray = arr
    ditherMatrix = GenerateDitherMatrix(n) * n ** 2
    print(ditherMatrix)
    
    for x in range(max_x):
        for y in range(max_y):
            i = x % n
            j = y % n
            if arr[x][y] > ditherMatrix[i][j] : 
                finalArray[x][y] = 1
            else:
                finalArray[x][y] = 0
    return finalArray

def main():
    fileName = './images/images (24).jpg'
    img = mpimg.imread(fileName,format='jpeg')

    arr = GrayScaleConverter2(img)

    # arbitrary initialize 
    n = 3
    while isPowerOfTwo(n) == False:
        n = int(input("Please Enter Size of Window :  "))

    finalArr = OrderedDithering(arr,n)

    plt.imshow(finalArr, cmap=plt.get_cmap('gray'))
    plt.show()


if __name__ == "__main__":
    main()