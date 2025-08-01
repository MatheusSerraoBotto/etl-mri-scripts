import cv2
import os

def show_images_with_filter(folder):
    image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()
    i = 0
    n = len(image_files)

    while i < n:
        filename = image_files[i]
        path = os.path.join(folder, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Erro ao carregar {filename}")
            i += 1
            continue

        discard = is_predominantly_dark(img)
        label = "DISCARDA" if discard else "MANTÉM"
        color = (0, 0, 255) if discard else (0, 255, 0)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Tamanho proporcional e posição dinâmica
        font_scale = 1.2
        thickness = 2
        position = (20, 50)

        cv2.putText(img_rgb, f"{label}", position,
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(img_rgb, filename, (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Visualizador", img_rgb)

        key = cv2.waitKey(0)

        if key == ord('q'):  # Sair
            break
        elif key == ord('d'):  # Avança 10 imagens
            i += 10
        else:  # Qualquer outra tecla: avança 1 imagem
            i += 1

    cv2.destroyAllWindows()

# Altere o caminho abaixo para sua pasta
if __name__ == "__main__":
    pasta = "output/slices/HR/sub"
    show_images_with_filter(pasta)