import pygame
from tkinter import messagebox,Tk
from PIL import Image
import numpy as np
import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

pygame.init()
screen = pygame.display.set_mode((1000,700))
pygame.display.set_caption("My first game")
clock = pygame.time.Clock()

model = tf.keras.models.load_model('model.h5')

loop = True
press = False
Pixels = []
pix = []
while loop:
    try:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                loop = False
        px, py = pygame.mouse.get_pos()
        if pygame.mouse.get_pressed() == (1,0,0):
            pygame.draw.rect(screen, (255,255,255), (px,py,30,30))
            
            
        if event.type == pygame.KEYDOWN:
            pygame.image.save(screen,'filena.png')

            image = Image.open('filena.png')
            new_image = image.resize((28, 28))
            new_image.save('fillna.png')

            img = mpimg.imread('fillna.png')
            gray = rgb2gray(img)

            image_resized = gray.reshape(1,28,28,1)


            pred = model.predict(image_resized)
            #print(np.round(pred[0],2))

            t = np.argmax(pred[0])
            print(pred.argmax())

            window = Tk()
            window.withdraw()
            messagebox.showinfo("Prediction is : " + str(t))
            window.destroy()

            
        if event.type == pygame.MOUSEBUTTONUP:
            press == True
        pygame.display.update()
        #clock.tick(1000)
    except Exception as e:
        print(e)
        pygame.quit()

pygame.quit()
