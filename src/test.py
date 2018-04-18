def test1():
    import cv2
    import environment
    import control

    env = environment.Environment(True)
    control = control

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    while(True):
        # Capture frame-by-frame
        ret, image = cap.read()
        if ret:
            # Grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Get env state
            (active, playarea, reward, done) = env.process(gray)
            print("Active:", active, "Reward:", reward, "Done", done)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # wait for 1 millisecond
            break

    cap.release()
    cv2.destroyAllWindows()


def test2():
    import control

    control = control.output


if __name__ == '__main__':
    test2()
