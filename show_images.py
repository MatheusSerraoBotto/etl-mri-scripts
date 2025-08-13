import nibabel as nib
import matplotlib.pyplot as plt

# === CONFIGURAR O PLANO DESEJADO AQUI: ===================
orientation = 'sagittal'   # ou 'coronal' ou 'sagittal'
# =========================================================

path = "ds006001/anat/sub-C1_acq_FLASH20_200um.nii.gz"
# path = "ds005533/sub-01/ses-nocap/anat/sub-01_ses-nocap_T1w.nii.gz"
# Voxel size (mm): (1.0, 1.0, 1.0)
# Image shape: (176, 256, 256)
# path = "ds002675/sub-c01/anat/sub-c01_T1w.nii.gz"
# Voxel size (mm): (1.0, 1.0, 1.0)
# Image shape: (176, 256, 256)
# path = "ds000244/sub-01/ses-00/anat/sub-01_ses-00_T1w.nii.gz"
# Voxel size (mm): (1.0, 1.0, 1.1)
# Image shape: (240, 256, 160)
# path = "ds000113/sub-01/ses-forrestgump/anat/sub-01_ses-forrestgump_T2w.nii.gz"
# Voxel size (mm): (0.6999283, 0.6666667, 0.6666667)
# Image shape: (274, 384, 384)
# path = "ds006001/anat/sub-C1_acq_FLASH20_200um.nii.gz"
# Voxel size (mm): (0.2, 0.2, 0.2)
# Image shape: (960, 840, 640)
img = nib.load(path)
data = img.get_fdata()

voxel_sizes = img.header.get_zooms()
print("Voxel size (mm):", voxel_sizes)
print("Image shape:", data.shape)

# Define função para obter slice de acordo com o plano
def get_slice(idx):
    if orientation == 'axial':
        return data[:, :, idx]
    elif orientation == 'coronal':
        return data[:, idx, :]
    elif orientation == 'sagittal':
        return data[idx, :, :]
    else:
        raise ValueError("Orientação inválida. Use 'axial', 'coronal' ou 'sagittal'.")

# Define número de fatias por plano
if orientation == 'axial':
    max_slices = data.shape[2]
elif orientation == 'coronal':
    max_slices = data.shape[1]
elif orientation == 'sagittal':
    max_slices = data.shape[0]

# Inicializa
current_slice = max_slices // 2
fig, ax = plt.subplots()
img_plot = ax.imshow(get_slice(current_slice), cmap='gray')
plt.title(f"{orientation.capitalize()} Slice {current_slice}/{max_slices-1}")
plt.axis('off')

# Atualiza imagem
def update_slice(new_idx):
    global current_slice
    current_slice = new_idx % max_slices
    img_plot.set_data(get_slice(current_slice))
    ax.set_title(f"{orientation.capitalize()} Slice {current_slice}/{max_slices-1}")
    fig.canvas.draw_idle()

# Navegação com setas
def on_key(event):
    if event.key == 'right':
        update_slice(current_slice + 1)
    elif event.key == 'left':
        update_slice(current_slice - 1)

fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()
