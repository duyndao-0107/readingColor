from collections import Counter
from sklearn.cluster import KMeans
from matplotlib import colors
import matplotlib.pyplot as plt
import cv2
from gtts import gTTS
import playsound
import os
import random

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()
while (cap.isOpened()):
    ret, fr = cap.read()
    if not ret:
        break
    w = int(fr.shape[1]*0.5)
    h = int(fr.shape[0]*0.5)
    fr = cv2.resize(fr, (w,h))
    img = cv2.flip(fr,1)
    cv2.imshow('Picture',img)
    if (cv2.waitKey(1) & 0xFF == 27):
        break
cap.release()
cv2.destroyAllWindows()

def preprocess(raw):
    img = cv2.resize(raw, (900, 600), interpolation = cv2.INTER_AREA)
    img = img.reshape(img.shape[0]*img.shape[1],3)
    return img
def rgb_to_hex(rgb_color):
    hex_color = "#"
    for i in rgb_color:
        hex_color += ("{:02x}".format(int(i)))
    return hex_color
def analyze(img):
    clf = KMeans(n_clusters = 5)
    color_labels = clf.fit_predict(img)
    center_colors = clf.cluster_centers_
    counts = Counter(color_labels)
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [rgb_to_hex(ordered_colors[i]) for i in counts.keys()]

    plt.figure(figsize = (12, 8))
    plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)

    plt.savefig("results/my_pie.png")
    print("Found the following colors:\n")
    for color in hex_colors:
        print(color)
        r1 = random.randint(1,10000000)
        randfile = "voice"+str(r1)+".mp3"
        myobj = gTTS(color, lang='en', slow=False)
        myobj.save(randfile)
        playsound.playsound(randfile)
        os.remove(randfile)
        
modified_image = preprocess(img)
analyze(modified_image)
