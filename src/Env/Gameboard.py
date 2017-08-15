import cv2
import numpy as np
from scipy.ndimage.measurements import label
# from skimage.measure import label

from scipy.ndimage.morphology import generate_binary_structure

from matplotlib import pyplot

class GameBoard():
    GAMEBOX = [[227, 51], [1048, 682]]

    def __init__(self):
        ''' init '''
        self.image_number = 0

    def saveImage(self, name, image):
        self.image_number += 1  
        print(self.image_number)
    
        # r,g,b = cv2.split(image)
        # bgrImage = cv2.merge([b,g,r])
        cv2.imwrite('../../output_images/{}_{}.png'.format(name, str(self.image_number).zfill(5)), image)


    def saveUnique(self, item, sprite):
        item = item.flatten()
        hsh = hash(item.tostring())

        if not hsh in self.sprites.keys():
            # print(hsh, full)
            # self.file.write("{}\n{}\n{}\n".format(hsh, full.flatten(), full))
            self.saveImage(hsh[-8], image)
            self.sprites[hsh] = item

    def draw_squares(self, image, squares):
        marked_image = np.copy(image)
        for square in self.squares:
            tl, br = square
            image = self.draw_squares(marked_image, tl, br)
        return marked_image

    def processImage(self, image):
        ''' init '''
        tl, br = self.GAMEBOX
        gameboard_image = image[tl[1]:br[1], tl[0]:br[0]]

        gray = cv2.cvtColor(gameboard_image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

        binary_image = np.zeros_like(thresh)
        binary_image[gray >= 1] = 255
        
        backtorgb = cv2.cvtColor(binary_image,cv2.COLOR_GRAY2RGB)


        # labels = label(binary_image, background=0) # skimage
        # print("count", np.max(np.unique(labels)))
        # for sprite in np.unique(labels):
        # labels, sprites_found = label(binary_image) # scipy

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1]
        for c in cnts:
            print(c)
            M = cv2.moments(c)
            if (M["m00"] > 0):
                cX = int((M["m10"] / M["m00"]))
                cY = int((M["m01"] / M["m00"]))
                cv2.rectangle(backtorgb, (cX, cY), (cX+10, cY+10), (255, 0, 0), 2)

        print(cnts)
        # print("count", sprites_found)


        # for sprite in range(1, sprites_found+1):
        #     if sprite == 0:
        #         continue

        #     nonzero = (labels[0] == sprite).nonzero()
        #     img = np.zeros_like(thresh)
        #     img[labels == sprite] = 255
        #     x = np.sum(img, axis=0)
        #     y = np.sum(img, axis=1)
        #     # Define a bounding box based on min/max x and y    
        #     bbox = ((self.masked_argmin(x), self.masked_argmin(y)), (np.argmax(x), np.argmax(y)))
            
        #     print(bbox, bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1])
        #     cv2.rectangle(backtorgb, bbox[0], bbox[1], (255, 0, 0), 2)

        cv2.imshow("game", backtorgb)
        # pyplot.imshow(backtorgb, cmap='gray')
        # pyplot.show()
        return gameboard_image

    def masked_argmin(self, a): # Defining func for regular array based soln
        b = np.zeros_like(a)
        b[a==0] = max(a)
        return np.argmin(b)

def testVideo(path):
    from moviepy.editor import VideoFileClip
    import os.path

    renv = GameBoard()

    filename, file_extension = os.path.splitext(path)
    output_path = "{}-processed{}".format(filename, file_extension)
    clip = VideoFileClip(path)
    processed_clip = clip.subclip(8).fl_image(renv.processImage)
    processed_clip.write_videofile(output_path, audio=False)


if __name__ == '__main__':
    testVideo('../../video/robotron.mp4')
