import cv2 as cv
import numpy as np

def faz_canny(quadro):
    # Converte o quadro para escala de cinza aplica desfoque gaussiano e detecta bordas usando canny
    cinza = cv.cvtColor(quadro, cv.COLOR_RGB2GRAY)
    desfoque = cv.GaussianBlur(cinza, (5, 5), 0)
    canny = cv.Canny(desfoque, 50, 150)
    return canny

def faz_segmentacao(quadro):
    # Usa as dimensoes do quadro para criar uma mascara triangular e manter apenas a regiao de interesse
    altura = quadro.shape[0]
    poligonos = np.array([
        [(0, altura), (800, altura), (380, 290)]
    ])
    mascara = np.zeros_like(quadro)
    cv.fillPoly(mascara, poligonos, 255)
    segmento = cv.bitwise_and(quadro, mascara)
    return segmento

def calcula_linhas(quadro, linhas):
    # Separa as linhas da esquerda e da direita calcula a media e obtem as coordenadas finais
    esquerda = []
    direita = []

    for linha in linhas:
        x1, y1, x2, y2 = linha.reshape(4)
        parametros = np.polyfit((x1, x2), (y1, y2), 1)
        inclinacao = parametros[0]
        intercepto_y = parametros[1]

        if inclinacao < 0:
            esquerda.append((inclinacao, intercepto_y))
        else:
            direita.append((inclinacao, intercepto_y))

    media_esquerda = np.average(esquerda, axis=0)
    media_direita = np.average(direita, axis=0)

    linha_esquerda = calcula_coordenadas(quadro, media_esquerda)
    linha_direita = calcula_coordenadas(quadro, media_direita)

    return np.array([linha_esquerda, linha_direita])

def calcula_coordenadas(quadro, parametros):
    # Calcula as coordenadas x e y da linha a partir da inclinacao e do intercepto
    inclinacao, intercepto = parametros
    y1 = quadro.shape[0]
    y2 = int(y1 - 150)
    x1 = int((y1 - intercepto) / inclinacao)
    x2 = int((y2 - intercepto) / inclinacao)
    return np.array([x1, y1, x2, y2])
