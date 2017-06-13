import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

def dispImg(image, n):
    return cv2.imshow('window'+str(n), image)

def greyscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def canny(image, l_threshold, h_threshold):
    return cv2.Canny(image,l_threshold, h_threshold)

def blur(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def roi(image):
    height = image.shape[0]
    width = image.shape[1]
    vertices = np.array( [[
                [3*width/4, 3*height/5],
                [width/4, 3*height/5],
                [40, height],
                [width - 40, height]
            ]], dtype=np.int32 )

    mask = np.zeros_like(image)
    #if len(img.shape) > 2:
    #    channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
    #    ignore_mask_color = (255,) * channel_count
    #else:
    #    ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(image, mask)
    return masked

def draw_lines(image, lines):
    try:
        for line in lines:
            coords = line[0]
            cv2.line(image, (coords[0],coords[1]), (coords[2],coords[3]), [255,255,255], 3)
    except:
        pass

def houghline(image, rho, theta, threshold, min_len, max_gap):
    lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength = min_len, maxLineGap = max_gap)
    lineImage= np.zeros((*image.shape, 3), dtype = np.uint8)
    draw_lines(lineImage, lines)
    return lineImage

def weighted_image(image, initial_image, a=0.8, b=1.0, c=0.0):
    return cv2.addWeighted(initial_image, a, image, b, c)

def process_image(image):
    imageBlurred = blur(image,11)
    #dispImg(imageBlurred, 1)

    imageEdges = canny(imageBlurred, 40, 50)
    #dispImg(imageEdges, 2)

    imageROI = roi(imageEdges)
    #dispImg(imageROI, 3)

    lineMarkedImage = houghline(imageROI, 1, np.pi/180, 40, 30, 200)
    #dispImg(lineMarkedImage, 4)   

    finalImage = weighted_image(lineMarkedImage, image)
    dispImg(finalImage, 5)

    return finalImage

def main():
    
    image = mpimg.imread('image3.jpg')
    outputImage = process_image(image)
    cv2.imwrite("outputImage3.jpg", outputImage)
    #white_output = 'outputVideo.mp4'
    #clip1 = VideoFileClip("video1.mp4")

    #white_clip = clip1.fl_image(process_image)
    #white_clip.write_videofile(white_output, audio=False)

if __name__ ==  '__main__':
    main()
