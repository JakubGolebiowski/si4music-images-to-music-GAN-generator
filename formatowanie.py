import cv2
import os
import numpy as np
# save numpy array as npz file
from numpy import asarray
from numpy import save

path = "D:\\semestr5\\SIwMuzyce\\trainData\\Ai\\Ai"
img = "\\img15x15"
midi = "\\midi18x18\\"
imgs = []
midis = []
data = []


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray = np.ceil(gray)
    return gray


def load_images_from_folder(folder1, folder2, imgs, midis, data):
    # images = []
    # numbers = []
    for filename in os.listdir(folder1):
        name = filename.split("-")
        mid = name[int(name[5][1]) - 1] + ".txt"
        with open(str(folder2) + str(mid), 'r') as f:
            load01 = [[int(num) for num in line.split(' ')] for line in f]
            temp1 = []
            for i in load01:
                temp2 = []
                for j in i:
                    temp2.append([j])
                temp1.append(temp2)
            midis.append(temp1)
        # midis.append(mid)

        img = cv2.imread(os.path.join(folder1, filename))
        # print(imgs)
        temp = []
        if img is not None:
            temp.append(rgb2gray(img))
        for i in temp:
            temp2 = []
            for j in i:
                for z in j:
                    temp2.append(z)
            imgs.append(temp2)

    # slownik = []
    # slownik.append([0, midis[0]])
    # iter = 1
    # for i in midis:
    #     j=0
    #     for z in slownik:
    #         if z[1] == i:
    #             j+=1
    #     if j ==0:
    #         slownik.append([iter, i])
    #         iter+=1
    #     else:
    #         continue
    # for i in midis:
    #     for j in slownik:
    #         if i == j[1]:
    #             numbers.append(j[0])
    #             break

    # print(numbers)
    # print(len(numbers))
    # print(len(slownik))
    print(len(midis))
    print(len(imgs))
    print(len(imgs[0]))
    # print(len(imgs[0][0]))
    # print(len(imgs[0][0]))

    # print(imgs[0], len(imgs))
    print(len(midis[0]))
    print(len(midis[0][0]))
    print(len(midis[0][0][0]))
    data.append(imgs)
    data.append(midis)
    # data.append(numbers)
    return data


data01 = load_images_from_folder(path + img, path + midi, imgs, midis, data)

# # define data
# data = asarray(data01[0])
# # save to npy file
# save('imgs7090x225.npy', data)
#
# data2 = asarray(data01[1])
# save('midis7090x18x18x1.npy', data2)

# numbers = asarray(data01[2])
# save('labels2.npy', numbers)
