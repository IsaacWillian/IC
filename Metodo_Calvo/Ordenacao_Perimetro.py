import skimage.draw as draw
import numpy as np

def ordenar_perimetro(rr,cc,row,col):
    baixo = []
    cima = []
    baixo_esquerda=[]
    baixo_direita = []
    cima_direita = []
    cima_esquerda = []
    new_rr = []
    new_cc = []

    listZip = zip(rr,cc)

    pontos = list(listZip)

    pontos = sorted(set(pontos))

    for ponto in pontos:
        if ponto[0] > row:
            baixo.append(ponto)
        else:
            cima.append(ponto)

    for ponto in baixo:
        if ponto[1] > col:
            baixo_direita.append(ponto)
        else:
            baixo_esquerda.append(ponto)

    for ponto in cima:
        if ponto[1] > col:
            cima_direita.append(ponto)
        else:
            cima_esquerda.append(ponto)

    cima_esquerda.sort(reverse=True)
    cima_esquerda.sort(key=lambda ponto:ponto[1])

    cima_direita.sort()
    cima_direita.sort(key=lambda ponto:ponto[1])

    baixo_direita.sort()
    baixo_direita.sort(key=lambda ponto:ponto[1],reverse=True)

    baixo_esquerda.sort(reverse=True)
    baixo_esquerda.sort(key=lambda ponto:ponto[1],reverse=True)

    cima_esquerda.extend(cima_direita)
    cima_esquerda.extend(baixo_direita)
    cima_esquerda.extend(baixo_esquerda)

    for ponto in cima_esquerda:
        new_rr.append(ponto[0])
        new_cc.append(ponto[1])

    array_rr = np.asarray(new_rr)
    array_cc = np.asarray(new_cc)

    array_rr = -(array_rr - row)
    array_cc = array_cc - col

    angles = np.arctan2(array_rr,array_cc)*180/np.pi
    angles = angles + 180

    return new_rr,new_cc,angles



