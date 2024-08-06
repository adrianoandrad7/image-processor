import sys
import os

# Adicione o diretório raiz ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np

def apply_frequency_filter(image, filter_type, cutoff):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = gray.shape
    crow, ccol = rows // 2 , cols // 2
    mask = np.ones((rows, cols, 2), np.uint8)
    if filter_type == 'low_pass':
        mask[crow-cutoff:crow+cutoff, ccol-cutoff:ccol+cutoff] = 1
    elif filter_type == 'high_pass':
        mask[crow-cutoff:crow+cutoff, ccol-cutoff:ccol+cutoff] = 0
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
    return img_back

def segment_object(imagem):
    
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    
    # Aplicar um limiar para segmentar os objetos
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Remover ruídos e pequenos objetos usando operações morfológicas
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    
    # Encontrar regiões desconhecidas
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marcar as regiões
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # Aplicar o algoritmo de watershed
    markers = cv2.watershed(imagem, markers)
    imagem[markers == -1] = [0, 0, 0]
    
    # Criar uma máscara para os objetos principais
    mask = np.zeros_like(imagem)
    mask[markers > 1] = [255, 255, 255]
    
    # Aplicar a máscara na imagem original para obter o resultado final com fundo negro
    result = cv2.bitwise_and(imagem, mask)
    
    return result



    

    


    





