import nibabel as nib
import matplotlib.pyplot as plt

# === CONFIGURAR O PLANO DESEJADO AQUI: ===================
orientation = 'sagittal'   # ou 'coronal' ou 'sagittal'
# =========================================================

# Carrega imagem
img = nib.load("ds006001/sub-C1/anat/sub-C1_acq_FLASH20_200um.nii.gz")
data = img.get_fdata()

voxel_sizes = img.header.get_zooms()
print("Voxel size (mm):", voxel_sizes)

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
