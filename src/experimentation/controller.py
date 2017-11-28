import serial
import time
import pygame

button_names = (
    "A Button",
    "B Button",
    "X Button",
    "Y Button",
    "Left Bumper",
    "Right Bumper",
    "Back",
    "Start",
    "Xbox Glowy Button",
    "Left Joystick Press",
    "Right Joystick Press",
)

axis_names = (
    # Axis Name,       (neg, pos)
    ("Left Joystick", ("Left", "Right")),
    ("Left Joystick", ("Up", "Down")),
    ("Left Trigger", ("Released", "Pressed")),
    ("Right Joystick", ("Left", "Right")),
    ("Right Joystick", ("Up", "Down")),
    ("Right Trigger", ("Released", "Pressed")),
    ("Digital dpad", ("Left", "Right")),
    ("Digital dpad", ("Up", "Down")),
)

pygame.init()
pygame.joystick.init()

joystick = pygame.joystick.Joystick(0)  
joystick.init()

print("Using joystick: ", joystick.get_name())

axes = joystick.get_numaxes()
buttons = joystick.get_numbuttons()
hats = joystick.get_numhats()

axes_pressed = [0] * axes
buttons_pressed = [0] * buttons
hats_pressed = [0] * hats

ser = None

def checkButtons(dontPrint=False):
    pygame.event.pump()
    for i in range( axes ):
        axis = joystick.get_axis( i )
        if (axes_pressed[i] != axis):
            #print("Axis {} changed from {} to {}".format(i, axes_pressed[i], axis))
            button_status = "zeroed"
            if axes < 0:
                button_status = axis_names[i][1][0]
            if axes > 0:
                button_status = axis_names[i][1][1]
            print("(Axis {0}) {1}: {2}".format(i, axis_names[i][0], button_status))
            axes_pressed[i] = axis

    for i in range( buttons ):
        button = joystick.get_button( i )
        if (buttons_pressed[i] != button):
            # print("Button {} changed from {} to {}".format(i, buttons_pressed[i], button))
            print("{0} {1}".format(button_names[i], "pressed" if button else "released"))
            buttons_pressed[i] = button

    for i in range( hats ):
        hat = joystick.get_hat( i )
        if (hats_pressed[i] != hat):
            print("Hat {} changed from {} to {}".format(i, hats_pressed[i], hat))
            hats_pressed[i] = hat

def writeThenRead(val):
    ser.write(bytes([val]))
    ser.flush()
    time.sleep(5)
    while ser.inWaiting():
        line = ser.readline()
        print("Read: ", line)
    time.sleep(1) 

def bitCheck():
    while ser.inWaiting():
        line = ser.readline()
        print("Read: ", line)

    for i in range(7):
        b = 1 << i
        print("{0}: {1} ({2:b})".format(i, bytes([b]), b))
        writeThenRead(b)
        checkButtons()
       

    for i in range(7):
        shift = 1 << 7
        b = (1 << i) | shift
        print("{0}: {1} ({2:b})".format(i, bytes([b]), b))
        writeThenRead(b)
        checkButtons()

def seqCheck():
    for i in range(25):
        writeThenRead(i)
        checkButtons()
        writeThenRead(255)
        checkButtons()

if __name__ == '__main__':

    checkButtons();

    ser = serial.Serial('/dev/ttyACM0')

    seqCheck()

    if ser is not None:
        ser.close()

    print("Done!")