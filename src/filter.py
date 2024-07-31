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

