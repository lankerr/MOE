import cv2
import numpy as np
import sys
sys.path.append("..")
from config import cfg
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from copy import deepcopy
from PIL import Image
import pandas as pd


def save_movie(data, save_path):
    seq_len, channels, height, width = data.shape
    data = data.transpose(0, 2, 3, 1)
    if data.dtype == cfg.data_type:
        data = (data * 255).astype(np.uint8)
    assert data.dtype == np.uint8
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # *'I420' *'PIM1' *'XVID' *'MJPG' 视频大小越来越小，都支持.avi
    writer = cv2.VideoWriter(save_path, fourcc, 1.0, (width, height))
    for i in range(seq_len):
        if cfg.dataset in ['human3.6m', 'ucf50', 'sports10', 'deformingthings4d']:
            color_data = data[i]
        else:
            color_data = cv2.cvtColor(data[i], cv2.COLOR_GRAY2BGR)
        writer.write(color_data)
    writer.release()


def save_image(data, save_path):
    data = data.transpose(0, 2, 3, 1)  # s h w c
    display_data = []
    if data.dtype == cfg.data_type:
        data = (data * 255.0).astype(np.uint8)
        # data = (data * 70).astype(np.uint8)
    assert data.dtype == np.uint8
    for i in range(data.shape[0]):
        if cfg.dataset in ['human3.6m', 'ucf50', 'sports10', 'deformingthings4d', 'jiangsu']:
            color_data = data[i]
        else:
            color_data = cv2.cvtColor(data[i], cv2.COLOR_GRAY2BGR)
        display_data.append(color_data)
    display_data = np.array(display_data)
    # max_value_1 = np.max(display_data[1, :, :,0])  # 假设前3个通道是图像数据
    # print(f"Max value in height*width is {max_value_1}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # for i in range(display_data.shape[0]):
    #     img = display_data[i,...,0]
    #     df = pd.DataFrame(img)
    #     df.to_excel(os.path.join(save_path, str(i + 1)+ '.xlsx'),index=False)
    for i in range(display_data.shape[0]):
        cv2.imwrite(os.path.join(save_path, str(i + 1) + '.png'), display_data[i, ..., 0])
        # img = Image.fromarray(display_data[i,...,0],mode='L')
        # img.save(os.path.join(save_path, f'image_{i+1}.png'))

# VIL_COLORS and VIL_LEVELS as defined previously
VIL_COLORS = [
    [0, 0, 0],
    [0.30196078431372547, 0.30196078431372547, 0.30196078431372547],
    [0.1568627450980392, 0.7450980392156863, 0.1568627450980392156863],
    [0.09803921568627451, 0.5882352941176471, 0.09803921568627451],
    [0.0392156862745098, 0.4117647058823529, 0.0392156862745098],
    [0.0392156862745098, 0.29411764705882354, 0.0392156862745098],
    [0.9607843137254902, 0.9607843137254902, 0.0],
    [0.9294117647058824, 0.6745098039215687, 0.0],
    [0.9411764705882353, 0.43137254901960786, 0.0],
    [0.6274509803921569, 0.0, 0.0],
    [0.9058823529411765, 0.0, 1.0]
]

VIL_LEVELS = [0, 16, 31, 59, 74, 100, 133, 160, 181, 219, 255]


def vil_cmap():
    cols = deepcopy(VIL_COLORS)
    lev = deepcopy(VIL_LEVELS)
    nil = cols.pop(0)
    under = cols[0]
    over = cols[-1]
    cmap = ListedColormap(cols)
    cmap.set_bad(nil)
    cmap.set_under(under)
    cmap.set_over(over)
    norm = BoundaryNorm(lev, len(cols))
    return cmap, norm


def plot_vil_sequences(input_seq, truth_seq, pred_seq, output_file):
    cmap, norm = vil_cmap()

    num_cols = max(input_seq.shape[0], truth_seq.shape[0], pred_seq.shape[0]) + 1
    fig, axes = plt.subplots(3, num_cols, figsize=(20, 15), gridspec_kw={'width_ratios': [1] * num_cols})

    sequences = [('Input Sequence', input_seq), ('Truth Sequence', truth_seq), ('Predicted Sequence', pred_seq)]

    for row, (label, seq) in enumerate(sequences):
        axes[row, 0].text(0.5, 0.5, label, fontsize=12, ha='center', va='center', rotation=90,
                          transform=axes[row, 0].transAxes)
        axes[row, 0].axis('off')

        for col in range(seq.shape[0]):
            im = seq[col, 0]  # Assuming the format (s, c, h, w) and c=1
            # im_standardized = (im / 20.3) * 255
            # im_standardized = np.clip(im_standardized, 0, 255).astype(np.uint8)
            im_standardized = (im / im.max()) * 255
            im_standardized = im_standardized.astype(np.uint8)
            img = axes[row, col + 1].imshow(im_standardized, cmap=cmap, norm=norm)
            axes[row, col + 1].axis('off')

    # Turn off any remaining axes
    for row in range(3):
        for col in range(seq.shape[0] + 1, num_cols):
            axes[row, col].axis('off')

    # Add a single colorbar on the right side
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(img, cax=cbar_ax, orientation='vertical')
    cbar.set_label('VIL Level')
    cbar.set_ticks(VIL_LEVELS)
    cbar.set_ticklabels([str(level) for level in VIL_LEVELS])

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
