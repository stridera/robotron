import numpy as np
import cv2
import time
from matplotlib import pyplot


GAMEBOXP = [[5.63885, 14.117647059], [1.221374046, 1.055718475]]

def dummyfun(x):
    return

def roi(img):
    imgShape = img.shape
    gb = np.array([
        [ imgShape[1] / GAMEBOXP[0][0], imgShape[0] / GAMEBOXP[0][1] ],
        [ imgShape[1] / GAMEBOXP[1][0], imgShape[0] / GAMEBOXP[1][1] ],
    ])

    vertices = np.array([[
        [ gb[0][0], gb[0][1] ],
        [ gb[1][0], gb[0][1] ],
        [ gb[1][0], gb[1][1] ],
        [ gb[0][0], gb[1][1] ],
    ]], np.int32)

    #blank mask:
    mask = np.zeros_like(img)

    # fill the mask
    cv2.fillPoly(mask, vertices, (255,255,255))
    
    # now only show the area that is the mask
    masked = cv2.bitwise_and(img, mask)

    return masked

def process_img(original_image, low_threshold, high_threshold):
    img_hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

    img_thresholded = cv2.inRange(img_hsv,low_threshold,high_threshold)
    erode_x = 2
    erode_y = erode_x
    dilate_x = 2
    dilate_y = dilate_x
    ekernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_x,erode_y))
    dkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_x,dilate_y))
    cv2.erode(img_thresholded,ekernel,img_thresholded,iterations = 1)
    cv2.dilate(img_thresholded,dkernel,img_thresholded,iterations = 1)

    processed_img = roi(img_thresholded)

    return processed_img


def findPlayer(screen):
    low_threshold = np.array([0, 0, 235])
    high_threshold = np.array([255, 20, 255])

    new_img_rgba = process_img(screen, low_threshold, high_threshold)
    new_img_rgba = roi(new_img_rgba)
    new_img_rgba = cv2.GaussianBlur(new_img_rgba,(9,9), 2,2)

    circles = cv2.HoughCircles(new_img_rgba, cv2.HOUGH_GRADIENT, 2, 23, param1=100, param2=20, minRadius=5, maxRadius=11)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(screen, (i[0], i[1]), i[2], (0, 255, 0), 2)  # draw the outer circle
            break
    return new_img_rgba, circles

def findEnemy(screen):
    low_threshold = np.array([0, 0, 20])
    high_threshold = np.array([0, 0, 160])

    new_img_rgba = process_img(screen, low_threshold, high_threshold)
    new_img_rgba = roi(new_img_rgba)
    new_img_rgba = cv2.GaussianBlur(new_img_rgba, (9,9), 2,2)

    circles = cv2.HoughCircles(new_img_rgba,cv2.HOUGH_GRADIENT,2,23, param1=100, param2=20, minRadius=5, maxRadius=11)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(screen, (i[0], i[1]), i[2], (0, 0, 255), 2)  # draw the outer circle

    return new_img_rgba, circles

def screen_record(screen):
    last_time = time.time()

    new_screen, enemies = findEnemy(screen)
    new_screen2, playerloc = findPlayer(screen)

    if enemies is not None and playerloc is not None:
        player_x = playerloc[0][0][0]
        player_y = playerloc[0][0][1]
        closest = 1000
        direction = None

        for enemy in enemies:
            dist = abs(round(int(enemy[0][0]) - player_x))
            direct = round(int(enemy[0][0])- player_x)
            if dist < closest:
                closest = dist
                direction = direct
    else:
        print("Nothing found")

    print('loop took {} seconds  -  Player: {}  Enemies: {}'.format(time.time() - last_time, playerloc, enemies))
    last_time = time.time()

    cv2.imshow('window', screen)
    return screen

def processVideoFrame(img):
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    screen_record(img_bgr)

def testImage(path):
    img_rgba = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    screen_record(img_rgba)
    while (True):
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

def testVideo(path):
    from moviepy.editor import VideoFileClip
    import os.path

    filename, file_extension = os.path.splitext(path)
    output_path = "{}-processed{}".format(filename, file_extension)
    clip = VideoFileClip(path)
    processed_clip = clip.subclip(38, 40).fl_image(screen_record)
    processed_clip.write_videofile(output_path, audio=False)


if __name__ == '__main__':
    # testVideo('../../resources/video/robotron.mp4')
    testImage('../../resources/images/figure_2.png')
