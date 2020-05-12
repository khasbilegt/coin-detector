import math

import cv2 as cv
import matplotlib.pyplot as plt


circleAreas = []
circleCenters = []
circles = {}
coins = {"500": 0, "100": 0, "50": 0, "10": 0, "5": 0, "1": 0}

original = cv.imread("coins.jpg")
img = original.copy()
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = cv.medianBlur(img, 5)

ret, thresh = cv.threshold(img, 50, 255, cv.THRESH_BINARY_INV)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
sorted_ctrs = sorted(contours, key=lambda ctr: cv.boundingRect(ctr)[0])

for i, ctr in enumerate(sorted_ctrs):
    epsilon = 0.001 * cv.arcLength(ctr, True)
    approx = cv.approxPolyDP(ctr, epsilon, True)

    if len(approx) > 80:
        cv.drawContours(original, approx, -1, (0, 0, 255), 3)
        (x, y), radius = cv.minEnclosingCircle(ctr)
        if thresh[int(y)][int(x)] != 0:
            area = int(math.pi * (radius ** 2))
            circles[radius] = (int(x), int(y))
            fontColor = (0, 0, 0)
            imgcenter = (int(x - 15), int(y - 10))
            font = cv.FONT_HERSHEY_SIMPLEX

            if area > 7500:
                coins["500"] += 1
                text = "500"
                fontColor = (255, 255, 255)
            elif 7500 > area >= 6300:
                coins["100"] += 1
                text = "100"
                fontColor = (0, 0, 255)
            elif 6300 > area >= 5500:
                coins["10"] += 1
                text = "10"
                fontColor = (255, 255, 88)
            elif 5500 > area >= 5000:
                coins["50"] += 1
                text = "50"
                fontColor = (255, 0, 120)
            elif 5000 > area >= 3800:
                coins["5"] += 1
                text = "5"
                fontColor = (0, 255, 0)
            elif area < 3800:
                coins["1"] += 1
                text = "1"
                fontColor = (88, 255, 255)
#                 cv.putText(original, str(text), imgcenter, font, 0.6, fontColor, 2)
#                 cv.putText(original, str("{}: {}".format(text, int(radius))), imgcenter, font, 0.6, fontColor, 2)
#                 cv.circle(original, (int(x), int(y)), int(radius), fontColor, 2)
#                 cv.rectangle(original, (int(x), int(y)), (int(x)+5,int(y)+5), fontColor, 5)

plt.title(
    "Detected coins | 500: {0}, 100: {1}, 50: {2}, 10: {3}, 1: {4}".format(
        coins["500"], coins["100"], coins["50"], coins["10"], coins["1"]
    )
)
plt.imshow(cv.cvtColor(original, cv.COLOR_BGR2RGB))
plt.show()
