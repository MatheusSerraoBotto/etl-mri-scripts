import nibabel as nib
import numpy as np
import cv2
import os

# ========== CONFIGURAÇÕES ==========
orientation = 'axial'  # 'axial', 'coronal', 'sagittal'
nii_path = "ds006001/sub-C1/anat/sub-C1_acq_FLASH20_200um.nii.gz"
output_dir = "slices_sr_test"
num_slices = 100
scale_factor = 4  # Redução para gerar a versão LR (ex: 2x, 4x, 8x)
normalize = True  # Normalizar para 0-255
to_uint8 = False   # Se False, salva em float32 (útil para treinar redes)

# ====================================

# Cria diretórios
hr_dir = os.path.join(output_dir, f"HR")
lr_dir = os.path.join(output_dir, f"LR_x{scale_factor}")
os.makedirs(hr_dir, exist_ok=True)
os.makedirs(lr_dir, exist_ok=True)

# Carrega imagem
img = nib.load(nii_path)
data = img.get_fdata()

# Checa resolução física dos voxels
voxel_sizes = img.header.get_zooms()
print(f"Voxel sizes (mm): {voxel_sizes}")
print(f"Original shape: {data.shape}")

# Normalização
if normalize:
    data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255.0

# Conversão de tipo
if to_uint8:
    data = data.astype(np.uint8)
else:
    data = data.astype(np.float32)

# Define plano de fatia
if orientation == 'axial':
    total_slices = data.shape[2]
    get_slice = lambda i: data[:, :, i]
elif orientation == 'coronal':
    total_slices = data.shape[1]
    get_slice = lambda i: data[:, i, :]
elif orientation == 'sagittal':
    total_slices = data.shape[0]
    get_slice = lambda i: data[i, :, :]
else:
    raise ValueError("Orientação inválida. Use 'axial', 'coronal' ou 'sagittal'.")

# Seleciona fatias do meio
start_idx = total_slices // 2 - num_slices // 2

# Salva fatias HR e LR
for i in range(num_slices):
    idx = start_idx + i
    slice_img = get_slice(idx)
    slice_img = np.rot90(slice_img)  # Rotaciona para visualização padrão (opcional)

    # Caminhos de saída
    hr_path = os.path.join(hr_dir, f"slice_{i:03d}.png")
    lr_path = os.path.join(lr_dir, f"slice_{i:03d}.png")

    # Salva HR
    cv2.imwrite(hr_path, slice_img)

    # Cria LR: downscale + upscale
    h, w = slice_img.shape
    low_res_size = (w // scale_factor, h // scale_factor)
    lr_img = cv2.resize(slice_img, low_res_size, interpolation=cv2.INTER_AREA)
    # lr_img = cv2.resize(lr_img, (w, h), interpolation=cv2.INTER_LINEAR)

    # Salva LR
    cv2.imwrite(lr_path, lr_img)

print(f"\n✅ {num_slices} slices salvos em HR e LR com fator {scale_factor}x.")
print(f"📂 HR: {hr_dir}")
print(f"📂 LR: {lr_dir}")
