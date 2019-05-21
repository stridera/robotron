import math

def getDirection(x2, y2, x1, y1):
    deltaX = x2 - x1
    deltaY = y2 - y1

    degrees_temp = (math.atan2(deltaY, deltaX)/math.pi*180)
    if degrees_temp < 0:
        degrees_final = 360 + degrees_temp
    else:
        degrees_final = degrees_temp

    point = round(degrees_final / 45) - 1

    print((x1, y1), (x2, y2), degrees_temp, degrees_final, point)

    if point > 8:
        point -= 8

    return point

getDirection(5, 5,  5,  0)
getDirection(5, 5,  7,  0)
getDirection(5, 5, 10,  0)
getDirection(5, 5, 10,  5)
getDirection(5, 5, 10, 10)
getDirection(5, 5,  5, 10)
getDirection(5, 5,  0, 10)
getDirection(5, 5,  0,  5)
getDirection(5, 5,  0,  0)
getDirection(5, 5,  3,  0)
getDirection(5, 5,  8,  0)
getDirection(5, 5,  5,  5)
