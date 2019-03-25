import numpy as np
import cv2
from matplotlib import pyplot
 
GAMEBOX = [[227, 51], [1048, 682]]
GAMEBOXP = [[5.63885, 14.117647059], [1.221374046, 1.055718475]]

def dummyfun(x):
    return
 
def roi(img, vertices):
    #blank mask:
    mask = np.zeros_like(img)
    # fill the mask
    cv2.fillPoly(mask, vertices, 255)
    # now only show the area that is the mask
    masked = cv2.bitwise_and(img, mask)
    return masked
 
def process_img(original_image, low_threshold, high_threshold):
    img_hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
 
    img_thresholded = cv2.inRange(img_hsv,low_threshold,high_threshold)
    imgShape = original_image.shape

    gb = np.array([
        [ imgShape[1] / GAMEBOXP[0][0], imgShape[0] / GAMEBOXP[0][1] ],
        [ imgShape[1] / GAMEBOXP[1][0], imgShape[0] / GAMEBOXP[1][1] ],
    ])

    vertices = np.array([
            [ gb[0][0], gb[0][1] ],
            [ gb[1][0], gb[0][1] ],
            [ gb[1][0], gb[1][1] ],
            [ gb[0][0], gb[1][1] ],
        ], np.int32)
    processed_img = roi(img_thresholded, [vertices])

    return processed_img
 
def findPlayer(screen):
    low_threshold = np.array([0, 0, 235])
    high_threshold = np.array([255, 20, 255])
    new_img_rgba = process_img(screen,low_threshold,high_threshold)
    
    oMoments = cv2.moments(new_img_rgba)
 
    dM01 = oMoments['m01']
    dM10 = oMoments['m10']
    dArea = oMoments['m00']
    posX=0
    posY=0
    if dArea > 10000:
        posX = int(dM10/dArea)
        posY = int(dM01/dArea)
    pt1 = (posX-10,posY-10)
    pt2 = (posX+10,posY+10)
    cv2.rectangle(screen,pt1,pt2,(0,255,0),2)
    return new_img_rgba
 
def findEnemy(screen):
    low_threshold = np.array([0, 0, 78])
    high_threshold = np.array([0, 0, 153])
    new_img_rgba = process_img(screen,low_threshold,high_threshold)
 
    oMoments = cv2.moments(new_img_rgba)
 
    dM01 = oMoments['m01']
    dM10 = oMoments['m10']
    dArea = oMoments['m00']
    posX=0
    posY=0
    if dArea > 10000:
        posX = int(dM10/dArea)
        posY = int(dM01/dArea)
    pt1 = (posX-10,posY-10)
    pt2 = (posX+10,posY+10)
    cv2.rectangle(screen,pt1,pt2,(0,0,255),2)
    return new_img_rgba
 
def main(img_rgba):
    print("Shape: {}".format(img_rgba.shape))
    while True:
        new_img_rgba = findPlayer(img_rgba)
        new_img_rgba2 = findEnemy(img_rgba)

        cv2.imshow('Player Threshold', new_img_rgba)
        cv2.imshow('Enemy Threshold', new_img_rgba2)

        cv2.imshow('window',img_rgba)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            print("Done.")
            cv2.destroyAllWindows()
            break

def testImage(path):
    img_rgba = cv2.imread('../../resources/images/figure_2.png',cv2.IMREAD_UNCHANGED)
    main(img_rgba)

def testVideo(path):
    from moviepy.editor import VideoFileClip
    import os.path

    filename, file_extension = os.path.splitext(path)
    output_path = "{}-processed{}".format(filename, file_extension)
    clip = VideoFileClip(path)
    processed_clip = clip.subclip(38).fl_image(main)
    processed_clip.write_videofile(output_path, audio=False)


if __name__ == '__main__':
    testVideo('../../resources/video/robotron.mp4')
    # main(cv2.imread('../../resources/images/figure_2.png',cv2.IMREAD_UNCHANGED)
