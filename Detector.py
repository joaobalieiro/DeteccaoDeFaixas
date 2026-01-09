import cv2 as cv
import numpy as np

def faz_canny(quadro):
    # Converte o quadro para escala de cinza aplica desfoque gaussiano e detecta bordas usando canny
    cinza = cv.cvtColor(quadro, cv.COLOR_RGB2GRAY)
    desfoque = cv.GaussianBlur(cinza, (5, 5), 0)
    canny = cv.Canny(desfoque, 50, 150)
    return canny

