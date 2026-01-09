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

def visualiza_linhas(quadro, linhas):
    # Cria uma imagem vazia e desenha as linhas detectadas
    linhas_visual = np.zeros_like(quadro)
    if linhas is not None:
        for x1, y1, x2, y2 in linhas:
            cv.line(linhas_visual, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return linhas_visual


# Le o video processa cada quadro detecta faixas e exibe o resultado
captura = cv.VideoCapture("rua.mp4")

while captura.isOpened():
    retorno, quadro = captura.read()

    canny = faz_canny(quadro)
    cv.imshow("canny", canny)

    segmento = faz_segmentacao(canny)

    hough = cv.HoughLinesP(
        segmento,
        2,
        np.pi / 180,
        100,
        np.array([]),
        minLineLength=100,
        maxLineGap=50
    )

    linhas = calcula_linhas(quadro, hough)

    linhas_visual = visualiza_linhas(quadro, linhas)
    cv.imshow("hough", linhas_visual)

    saida = cv.addWeighted(quadro, 0.9, linhas_visual, 1, 1)
    cv.imshow("output", saida)

    if cv.waitKey(10) & 0xFF == ord('q'):
        break

captura.release()
cv.destroyAllWindows()
