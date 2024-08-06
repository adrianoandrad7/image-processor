import sys
import os

# Adicione o diretório 'src' ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import matplotlib.pyplot as plt
from filter import apply_frequency_filter  
from filter import segment_object

def main():
    
    # 01 low_pass_filtered and high_pass_filtered
    image_path = ''
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erro ao carregar a imagem {image_path}")
        return
    low_pass_filtered = apply_frequency_filter(image, 'low_pass', 30)
    high_pass_filtered = apply_frequency_filter(image, 'high_pass', 30)
    plt.figure(figsize=(12, 6))
    plt.subplot(131), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original')
    plt.subplot(132), plt.imshow(low_pass_filtered, cmap='gray'), plt.title('Filtro Passa Baixa')
    plt.subplot(133), plt.imshow(high_pass_filtered, cmap='gray'), plt.title('Filtro Passa Alta')
    plt.show()
    
    
    # 02  Segmentar objeto
    image = cv2.imread(image_path)
    resultado = segment_object(image)
    
    # Salvar o resultado
    output_path = ''
    cv2.imwrite(output_path, resultado)
    
    
if __name__ == "__main__":
    main()
