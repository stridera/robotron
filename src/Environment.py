import cv2
import numpy as np

from ScoreProcessor import ScoreProcessor

class Environment():

    def __init__(self):
        ''' constructor '''
        self.ScoreProcessor = ScoreProcessor()

        self.score = 0;
        

        # Debug Stuff
        self.image_number = 0;

    def saveImage(self, image):
        self.image_number += 1  
        print(self.image_number)
    
        # r,g,b = cv2.split(image)
        # bgrImage = cv2.merge([b,g,r])
        cv2.imwrite('../output_images/blah{}.png'.format(str(self.image_number).zfill(5)), image)

    def overlayText(self, image, text, location, size=3, weight=8, color=(255,255,255)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, text, location, font, size, color, weight)
        return image

    def processImage(self, image):

        self.score = self.ScoreProcessor.getScore(image)
        image = self.overlayText(image, "Score: {}".format(self.score), (5, 40), weight=4, size=1)

        # pyplot.imshow(image)
        # pyplot.show()

        return image





    def post(self):
        print(len(self.numarr))
        with open('../data/numbers.dat', "wb") as f:
            pickle.dump(self.numarr, f)




def testVideo(path):
    from moviepy.editor import VideoFileClip
    import os.path

    renv = Environment()

    filename, file_extension = os.path.splitext(path)
    output_path = "{}-processed{}".format(filename, file_extension)
    clip = VideoFileClip(path)
    processed_clip = clip.subclip(8).fl_image(renv.processImage)
    processed_clip.write_videofile(output_path, audio=False)
    # renv.post()


if __name__ == '__main__':
    testVideo('../video/robotron.mp4')
