from tkinter import *
from tkinter import filedialog
import pygame
import os, random
from PIL import ImageTk, Image, ImageDraw
from threading import Timer
import math
from xml.etree import ElementTree
import sqlite3
import tensorflow as tf
import numpy as np
from skimage.transform import resize
from mido import MidiFile, Message, MetaMessage, MidiTrack, merge_tracks
from keras.models import load_model


# xml configuration
config_file_name = 'config.xml'
config_file_path = os.path.abspath(config_file_name)
dom = ElementTree.parse(config_file_path)
root = dom.getroot()
databaseName = root[2].text

# db setup
database_connection = sqlite3.connect(databaseName)
c = database_connection.cursor()
c.execute("""
    CREATE TABLE IF NOT EXISTS data (
    sound1 text,
    sound2 text,
    sound3 text,
    sound4 text,
    sound5 text,
    img1 text,
    img2 text,
    img3 text,
    img4 text,
    img5 text
    )""")

def rgb2gray(rgb):
  r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
  gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
  gray = np.ceil(gray)
  return gray


def grayImgToVector(grayImg):
  temp = []
  for j in grayImg:
    for z in j:
      temp.append(z)
  return temp

def MatrixToMidi(input, output):
    ss = input
    # ss = np.loadtxt(input, dtype=np.int)
    mid = MidiFile()
    tr = MidiTrack()
    curtime = 0
    maxtimepos = 0
    maxnotepos = 0
    print(ss.shape)
    print(ss)

    for i in range(18):
        if ss[0][17 - i] > 0:
            maxtimepos = 17 - i
            break
    for i in range(18):
        if ss[17 - i][0] > 0:
            maxnotepos = 17 - i
            break
    tr.append(Message('program_change', program=0, channel=0, time=0))
    for i in range(maxtimepos):
        for j in range(maxnotepos):
            if ss[j + 1][i + 1] != 0:
                if ss[j + 1][i + 1] > 0:
                    tr.append(Message(type='note_on', note=ss[j + 1][0], channel=0, velocity=ss[j + 1][i + 1],
                                      time=0 + ss[0][i + 1] - curtime))
                    curtime = ss[0][i + 1]
                else:
                    tr.append(Message(type='note_off', note=ss[j + 1][0], channel=0, velocity=-ss[j + 1][i + 1],
                                      time=0 + ss[0][i + 1] - curtime))
                    curtime = ss[0][i + 1]
    tr.append(MetaMessage('end_of_track', time=960 - curtime))

    mid.tracks.append(tr)

    mid.save(output)

    return mid


def get_all_rows_from_db():
    sqlite_select_query = """SELECT * from data"""
    c.execute(sqlite_select_query)
    return c.fetchall()


def get_image_names_from_record(row):
    image_file_names = []
    for i in range(5, 10): # img names starts from row[5]-row[9]
        image_file_names.append(row[i])
    return image_file_names


def load_image_from_file(image_file_name):
    image_path = "C:/Users/gancu/Desktop/SEM5/AIMUZYKA/KońcowaRozbudowa/DrawMusic/BazaDanychObrazy/" + image_file_name
    train_image = tf.keras.preprocessing.image.load_img(
        image_path, color_mode='grayscale', target_size=None,
        interpolation='nearest'
    )
    return train_image


def normalize_image(image):
    image = np.array(image)
    image = image / 255.0
    image = tf.reshape(image, [1, 480, 640, 1])
    return image


def flatten_image(image):
    initializer = 'ones'
    max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
    image = tf.keras.layers.Conv2D(1, 6, activation='relu', kernel_initializer=initializer,
                                         input_shape=(480, 640, 1))(image)
    image = tf.keras.layers.Conv2D(1, 6, activation='relu', kernel_initializer=initializer,
                                   input_shape=(480, 640, 1))(image)
    image = tf.keras.layers.Conv2D(1, 6, activation='relu', kernel_initializer=initializer,
                                   input_shape=(480, 640, 1))(image)
    image = tf.keras.layers.Conv2D(1, 6, activation='relu', kernel_initializer=initializer,
                                   input_shape=(480, 640, 1))(image)
    image = tf.keras.layers.Conv2D(1, 6, activation='relu', kernel_initializer=initializer,
                                   input_shape=(480, 640, 1))(image)
    image = max_pool_2d(image)
    image = tf.keras.layers.Conv2D(1, 6, activation='relu', kernel_initializer=initializer,
                                         input_shape=(480, 640, 1))(image)
    image = tf.keras.layers.Conv2D(1, 6, activation='relu', kernel_initializer=initializer,
                                   input_shape=(480, 640, 1))(image)
    image = tf.keras.layers.Conv2D(1, 6, activation='relu', kernel_initializer=initializer,
                                   input_shape=(480, 640, 1))(image)
    image = tf.keras.layers.Conv2D(1, 6, activation='relu', kernel_initializer=initializer,
                                   input_shape=(480, 640, 1))(image)

    image = max_pool_2d(image)
    image = tf.keras.layers.Conv2D(1, 6, activation='relu', kernel_initializer=initializer,
                                         input_shape=(480, 640, 1))(image)
    image = tf.keras.layers.Conv2D(1, 6, activation='relu', kernel_initializer=initializer,
                                   input_shape=(480, 640, 1))(image)
    image = tf.keras.layers.Conv2D(1, 6, activation='relu', kernel_initializer=initializer,
                                   input_shape=(480, 640, 1))(image)
    image = tf.keras.layers.Conv2D(1, 6, activation='relu', kernel_initializer=initializer,
                                   input_shape=(480, 640, 1))(image)
    image = tf.keras.layers.Conv2D(1, 6, activation='relu', kernel_initializer=initializer,
                                   input_shape=(480, 640, 1))(image)
    image = max_pool_2d(image)
    image = tf.keras.layers.Conv2D(1, 6, activation='relu', kernel_initializer=initializer,
                                         input_shape=(480, 640, 1))(image)
    image = tf.keras.layers.Conv2D(1, 6, activation='relu', kernel_initializer=initializer,
                                   input_shape=(480, 640, 1))(image)
    image = tf.keras.layers.Conv2D(1, 6, activation='relu', kernel_initializer=initializer,
                                   input_shape=(480, 640, 1))(image)

    image = max_pool_2d(image)
    image = tf.reshape(image, image.shape[1:3])

    image = (image / pow(10,25) / 28.651188)


    return image


def save_image_snap(train_image, file_name):
    train_image = train_image * 255
    arr = np.array(train_image)
    im = Image.fromarray(arr)
    im = im.convert('RGB')
    im.save('C:/Users/gancu/Desktop/SEM5/AIMUZYKA/KońcowaRozbudowa/DrawMusic/tensory/' + file_name)


def convert_image_to_train_image(image):
    train_image = normalize_image(image)
    train_image = flatten_image(train_image)
    train_image = resize(train_image, (15,15))
    # train_image = np.resize(train_image,(28,28))
    return train_image

def convert_image_to_predict(image):
    train_image = normalize_image(image)
    train_image = flatten_image(train_image)
    train_image = resize(train_image, (15, 15))
    train_image = train_image * 255
    # train_image = np.resize(train_image,(28,28))
    return train_image

def prepare_train_images():
    records = get_all_rows_from_db()
    current_record = 0
    for row in records:
        image_file_names = get_image_names_from_record(row)
        for i in range(5):
            image = load_image_from_file(image_file_names[i])
            train_image = convert_image_to_train_image(image)
            save_image_snap(train_image, image_file_names[i])
        current_record += 1
        print(current_record, "/", len(records))


def predict_and_postprocess(modelName, imgVector225, cut=20):
    model = modelName # make it global for fast use
    vec = imgVector225
    vec = np.array(vec).reshape(1, 15 * 15)
    vec = (vec.astype(np.float32)) / 255

    X = model.predict(vec)

    Y = X
    max = np.max(Y)
    min = np.min(Y)
    Y = np.array([(x - min) / (max - min) for x in Y]) * 2 - 1

    Y = Y * 544 + 544 - 128  # ((X_train.astype(np.float32) + 128) - 544 ) / 544
    Y = Y.astype(np.int16)
    # removes values between "cut" and -"cut"
    Y[(Y < cut) & (Y > -1 * cut)] = 0

    # zerowanie kolumn z ujemnym czasem
    for i in range(1, 18):
        if Y[0][0][i][0] <= 0:
            Y[0][0][i][0] = 0
            for j in range(1, 18):
                Y[0][j][i][0] = 0

    Y = Y.transpose()

    # sortowanie kolunm po czasie
    temp = []
    temp.append([Y[0][0]])
    b = []
    for i in range(1, 18):
        if Y[0, i, 0, 0] != 0:
            b.append([i, Y[0, i, 0, 0]])
    b = np.array(b)
    b = b[b[:, 1].argsort()]
    for el in b:
        temp[0].append(Y[0][el[0]])
    for i in range(len(temp[0]), 18):
        temp[0].append([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
    temp = np.array(temp)

    # temp
    temp = temp.astype('float64')
    for i in range(0, 18):
        for j in range(1, 18):
            if temp[0][i][j][0] != 0.0:
                temp[0][i][j][0] = (temp[0][i][j][0] + 128 - 554) / 554
                if i != 0:
                    temp[0][i][j][0] = (((temp[0][i][j][0] + 1) / 2) * 254) - 127
    temp[0][0] = np.abs(temp[0][0]) * 87 + 21
    temp[0][0][0][0] = 0.0
    temp = temp.astype(np.int16)

    temp = temp.transpose()

    return temp

def MergeMidiList(input): #List of MidiFiles to merge
    NewMidi = MidiFile();
    NewTrack = MidiTrack();
    for i in range(len(input)):
        for track in input[i].tracks:
            for msg in track:
                NewTrack.append(msg);
    NewMidi.tracks.append(NewTrack);
    return NewMidi

class DrawMusic(object):
    DEFAULT_PEN_SIZE = 8.0
    DEFAULT_COLOR = '#000000'

    def __init__(self):
        self.root = Tk()

        self.good_button = Button(self.root, text='good<Q>', command=self.good_samples)
        self.good_button.grid(row=0, column=0)

        self.images_directory_button = Button(self.root, text='choseDirectoryForImages',
                                              command=self.chose_directory_for_images)
        self.images_directory_button.grid(row=0, column=2)

        self.music_directory_button = Button(self.root, text='choseDirectoryForMusic',
                                             command=self.chose_directory_for_music)
        self.music_directory_button.grid(row=0, column=3)

        self.bad_button = Button(self.root, text='bad<E>', command=self.bad_samples)
        self.bad_button.grid(row=0, column=1)

        self.start_button = Button(self.root, text='START', command=self.start_music_playing)
        self.start_button.grid(row=0, column=4)

        self.prepare_images_button = Button(self.root, text='Prepare Images', command=prepare_train_images)
        self.prepare_images_button.grid(row=1, column=0)

        self.create_midi_button = Button(self.root, text='Convert to midi', state=DISABLED ,command=self.create_midi_from_images)
        self.create_midi_button.grid(row=1, column=1)

        self.toggle_button = Button(self.root, text='Change mode', command=self.toggle)
        self.toggle_button.grid(row=1, column=2)

        self.PILImage1 = Image.new("RGB", (640, 480), color=(255, 255, 255))
        self.PILDraw1 = ImageDraw.Draw(self.PILImage1)

        self.PILImage2 = Image.new("RGB", (640, 480), color=(255, 255, 255))
        self.PILDraw2 = ImageDraw.Draw(self.PILImage2)

        self.PILImage3 = Image.new("RGB", (640, 480), color=(255, 255, 255))
        self.PILDraw3 = ImageDraw.Draw(self.PILImage3)

        self.PILImage4 = Image.new("RGB", (640, 480), color=(255, 255, 255))
        self.PILDraw4 = ImageDraw.Draw(self.PILImage4)

        self.PILImage5 = Image.new("RGB", (640, 480), color=(255, 255, 255))
        self.PILDraw5 = ImageDraw.Draw(self.PILImage5)

        self.c = Canvas(self.root, bg='white', width=640, height=480)
        self.c.grid(row=2, columnspan=5)
        self.c1 = Canvas(self.root, bg='white', width=160, height=120)
        self.c1.grid(row=3, columnspan=1, column=0)
        self.c2 = Canvas(self.root, bg='white', width=160, height=120)
        self.c2.grid(row=3, columnspan=1, column=1)
        self.c3 = Canvas(self.root, bg='white', width=160, height=120)
        self.c3.grid(row=3, columnspan=1, column=2)
        self.c4 = Canvas(self.root, bg='white', width=160, height=120)
        self.c4.grid(row=3, columnspan=1, column=3)
        self.c5 = Canvas(self.root, bg='white', width=160, height=120)
        self.c5.grid(row=3, columnspan=1, column=4)

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.is_train_mode = 1

        self.old_x = 0
        self.old_y = 0
        self.line_width = self.DEFAULT_PEN_SIZE
        self.color = self.DEFAULT_COLOR
        self.c.bind('<B1-Motion>', self.pen_on)
        self.c.bind('<ButtonRelease-1>', self.pen_off)
        self.root.bind('<Key>', self.key_chose)

        self.timeStart = 0
        self.timeEnd = 0
        self.numberOfPoints = 0
        self.lastFiveCanvas = []
        self.musicIsPlaying = 0

        self.snapNumber = 1
        self.music_directory = root[0].text
        self.images_directory = root[1].text

    def toggle(self):
        if self.toggle_button.config('relief')[-1] == 'sunken':
            self.toggle_button.config(relief="raised")
            self.bad_button.config(state=NORMAL)
            self.good_button.config(state=NORMAL)
            self.images_directory_button.config(state=NORMAL)
            self.music_directory_button.config(state=NORMAL)
            self.prepare_images_button.config(state=NORMAL)
            self.create_midi_button.config(state=DISABLED)
            self.is_train_mode = 1

        else:
            self.toggle_button.config(relief="sunken")
            self.bad_button.config(state=DISABLED)
            self.good_button.config(state=DISABLED)
            self.images_directory_button.config(state=DISABLED)
            self.music_directory_button.config(state=DISABLED)
            self.prepare_images_button.config(state=DISABLED)
            self.create_midi_button.config(state=NORMAL)
            self.is_train_mode = 0


    def create_midi_from_images(self):
        model = load_model('D:/semestr5/SIwMuzyce/trainData/Ai/Ai/generator_model_final_100K.h5')
        mid = []
        tab = [self.PILImage1, self.PILImage2, self.PILImage3, self.PILImage4, self.PILImage5]
        for i in range(5):
            pix = np.array(tab[i])
            pix = rgb2gray(pix)
            newpix = convert_image_to_predict(pix)
            vector = grayImgToVector(newpix)
            vector = np.round(vector)

            midi = predict_and_postprocess(model, np.round(vector), 20)
            midi2 = np.reshape(midi, (18, 18))
            MatrixToMidi(midi2, "testMidi"+str(i)+".mid")

            mid.append(MidiFile("testMidi"+str(i)+".mid"))

        mid_5 = MergeMidiList(mid)
        mid_5.save("midi5.mid")


    def clearAll(self):
        self.c.delete("all")
        self.c1.delete("all")
        self.c2.delete("all")
        self.c3.delete("all")
        self.c4.delete("all")
        self.c5.delete("all")
        self.clearPIL()
        self.snapNumber = 1


    def clearPIL(self):
        self.PILDraw1.rectangle((0, 0, 640, 480), fill=(255, 255, 255))
        self.PILDraw2.rectangle((0, 0, 640, 480), fill=(255, 255, 255))
        self.PILDraw3.rectangle((0, 0, 640, 480), fill=(255, 255, 255))
        self.PILDraw4.rectangle((0, 0, 640, 480), fill=(255, 255, 255))
        self.PILDraw5.rectangle((0, 0, 640, 480), fill=(255, 255, 255))


    def makeSnap(self):
        if (self.snapNumber == 1):
            self.img = self.PILImage1.resize((160, 120), Image.ANTIALIAS)
            self.my_img1 = ImageTk.PhotoImage(self.img)
            self.c1.create_image((0, 0), anchor=NW, image=self.my_img1)
        elif self.snapNumber == 2:
            self.img = self.PILImage2.resize((160, 120), Image.ANTIALIAS)
            self.my_img2 = ImageTk.PhotoImage(self.img)
            self.c2.create_image((0, 0), anchor=NW, image=self.my_img2)
        elif self.snapNumber == 3:
            self.img = self.PILImage3.resize((160, 120), Image.ANTIALIAS)
            self.my_img3 = ImageTk.PhotoImage(self.img)
            self.c3.create_image((0, 0), anchor=NW, image=self.my_img3)
        elif self.snapNumber == 4:
            self.img = self.PILImage4.resize((160, 120), Image.ANTIALIAS)
            self.my_img4 = ImageTk.PhotoImage(self.img)
            self.c4.create_image((0, 0), anchor=NW, image=self.my_img4)
        elif self.snapNumber == 5:
            self.img = self.PILImage5.resize((160, 120), Image.ANTIALIAS)
            self.my_img5 = ImageTk.PhotoImage(self.img)
            self.c5.create_image((0, 0), anchor=NW, image=self.my_img5)

        self.snapNumber += 1
        self.clear_canvas()

    def good_samples(self):
        self.save_samples_in_db()
        self.clearAll()
        self.music()

    def bad_samples(self):
        self.clearAll()
        self.start_music_playing()

    def chose_directory_for_images(self):
        self.images_directory = filedialog.askdirectory()
        print("Wybrana lokalizacja zdjęć: ", self.images_directory)

    def chose_directory_for_music(self):
        self.music_directory = filedialog.askdirectory()
        print("Wybrana lokalizacja muzyki: ", self.music_directory)

    def save_samples_in_db(self):
        nameList = self.filename.split("-")
        imageName1 = self.filename.replace(".mid", "#") + "1" + ".png"
        imageName2 = self.filename.replace(".mid", "#") + "2" + ".png"
        imageName3 = self.filename.replace(".mid", "#") + "3" + ".png"
        imageName4 = self.filename.replace(".mid", "#") + "4" + ".png"
        imageName5 = self.filename.replace(".mid", "#") + "5" + ".png"
        self.PILImage1.save(self.images_directory + "/" + imageName1)
        self.PILImage2.save(self.images_directory + "/" + imageName2)
        self.PILImage3.save(self.images_directory + "/" + imageName3)
        self.PILImage4.save(self.images_directory + "/" + imageName4)
        self.PILImage5.save(self.images_directory + "/" + imageName5)
        c.execute(
            f"INSERT INTO data VALUES ('{nameList[0]}','{nameList[1]}','{nameList[2]}','{nameList[3]}','{nameList[4]}','{imageName1}','{imageName2}','{imageName3}','{imageName4}','{imageName5}')")

    def start_music_playing(self):
        self.clearAll()

        if(self.is_train_mode):
            t1 = Timer(1, self.makeSnap)
            t2 = Timer(2, self.makeSnap)
            t3 = Timer(3, self.makeSnap)
            t4 = Timer(4, self.makeSnap)
            t5 = Timer(5, self.makeSnap)
            t1.start()
            t2.start()
            t3.start()
            t4.start()
            t5.start()

            pygame.mixer.pre_init(44100, -16, 2, 1024)
            pygame.mixer.init()
            pygame.init()
            self.filename = str(random.choice(os.listdir(self.music_directory)))
            pygame.mixer.music.load(self.music_directory + "/" + (self.filename))
            pygame.mixer.music.play()
            self.musicIsPlaying = 1
        else:
            t1 = Timer(1, self.makeSnap)
            t2 = Timer(2, self.makeSnap)
            t3 = Timer(3, self.makeSnap)
            t4 = Timer(4, self.makeSnap)
            t5 = Timer(5, self.makeSnap)
            t1.start()
            t2.start()
            t3.start()
            t4.start()
            t5.start()


    def key_chose(self, event):
        if event.char == 'q':
            self.good_samples()
        elif event.char == 'e':
            self.bad_samples()

    def activate_button(self, some_button):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button

    def clear_canvas(self):
        self.c.delete("all")

    def pen_on(self, event):
        r, g, b = self.calculate_speed_color(event)
        speedColor = "#" + r + g + b

        if (self.is_train_mode):
            if (pygame.mixer.music.get_pos() != -1):
                paint_color = speedColor
                if self.old_x and self.old_y:
                    self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                                       width=self.line_width, fill=paint_color,
                                       capstyle=ROUND, smooth=TRUE, splinesteps=36)
                    if (self.snapNumber == 1):
                        self.PILDraw1.line([self.old_x, self.old_y, event.x, event.y], width=int(self.line_width),
                                           fill=paint_color)
                    elif (self.snapNumber == 2):
                        self.PILDraw2.line([self.old_x, self.old_y, event.x, event.y], width=int(self.line_width),
                                           fill=paint_color)
                    elif (self.snapNumber == 3):
                        self.PILDraw3.line([self.old_x, self.old_y, event.x, event.y], width=int(self.line_width),
                                           fill=paint_color)
                    elif (self.snapNumber == 4):
                        self.PILDraw4.line([self.old_x, self.old_y, event.x, event.y], width=int(self.line_width),
                                           fill=paint_color)
                    elif (self.snapNumber == 5):
                        self.PILDraw5.line([self.old_x, self.old_y, event.x, event.y], width=int(self.line_width),
                                           fill=paint_color)
                self.old_x = event.x
                self.old_y = event.y
        else:
            paint_color = speedColor
            if self.old_x and self.old_y:
                self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                                   width=self.line_width, fill=paint_color,
                                   capstyle=ROUND, smooth=TRUE, splinesteps=36)
                if (self.snapNumber == 1):
                    self.PILDraw1.line([self.old_x, self.old_y, event.x, event.y], width=int(self.line_width),
                                       fill=paint_color)
                elif (self.snapNumber == 2):
                    self.PILDraw2.line([self.old_x, self.old_y, event.x, event.y], width=int(self.line_width),
                                       fill=paint_color)
                elif (self.snapNumber == 3):
                    self.PILDraw3.line([self.old_x, self.old_y, event.x, event.y], width=int(self.line_width),
                                       fill=paint_color)
                elif (self.snapNumber == 4):
                    self.PILDraw4.line([self.old_x, self.old_y, event.x, event.y], width=int(self.line_width),
                                       fill=paint_color)
                elif (self.snapNumber == 5):
                    self.PILDraw5.line([self.old_x, self.old_y, event.x, event.y], width=int(self.line_width),
                                       fill=paint_color)
            self.old_x = event.x
            self.old_y = event.y


    def calculate_speed_color(self, event):
        dx = abs(int(self.old_x) - event.x)
        dy = abs(int(self.old_y) - event.y)
        dist = math.sqrt(pow(dx, 2) + pow(dy, 2)) * 3
        if (dist < 45):
            self.red = int(dist * 5)
            self.green = int(dist * 5)
            self.blue = int(dist * 5)
        else:
            self.red = 230
            self.green = 230
            self.blue = 230
        r = padded_hex(self.red)
        g = padded_hex(self.green)
        b = padded_hex(self.blue)
        return r, g, b

    def pen_off(self, event):
        self.old_x, self.old_y = 0, 0


def padded_hex(i):
    given_int = i
    given_len = 2

    hex_result = hex(given_int)[2:]  # remove '0x' from beginning of str
    num_hex_chars = len(hex_result)
    extra_zeros = '0' * (given_len - num_hex_chars)  # may not get used..

    return (hex_result if num_hex_chars == given_len else
            '?' * given_len if num_hex_chars > given_len else
            extra_zeros + hex_result if num_hex_chars < given_len else
            None)


if __name__ == '__main__':
    DrawMusic()
    database_connection.commit()
    database_connection.close()
