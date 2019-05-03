import math

def getDirection(x2, y2, x1, y1):
    deltaX = x2 - x1
    deltaY = y2 - y1

    degrees_temp = (math.atan2(deltaY, deltaX)/math.pi*180)
    if degrees_temp < 0:
        degrees_final = 360 + degrees_temp
    else:
        degrees_final = degrees_temp

    print(degrees_temp, degrees_final)

    point = round(degrees_final / 45) + 7
    if point > 8:
        point -= 8

    return point

print(1, getDirection(5, 5,  5,  0))
print(2, getDirection(5, 5, 10,  0))
print(3, getDirection(5, 5, 10,  5))
print(4, getDirection(5, 5, 10, 10))
print(5, getDirection(5, 5,  5, 10))
print(6, getDirection(5, 5,  0, 10))
print(7, getDirection(5, 5,  0,  5))
print(8, getDirection(5, 5,  0,  0))
print(9, getDirection(5, 5,  3,  0))
print(0, getDirection(5, 5,  8,  0))
print(1, getDirection(5, 5,  5,  5))
