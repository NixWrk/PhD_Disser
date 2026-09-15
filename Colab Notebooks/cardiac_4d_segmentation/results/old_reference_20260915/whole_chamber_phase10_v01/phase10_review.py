from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def load_volume(path):
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    spacing = np.linalg.norm(image.affine[:3, :3], axis=0)
    return image, data, spacing


def hu_image(data):
    return np.clip(np.asarray(data, dtype=float), -1000, 1200)


def montage(data, slices, plane, out_path, title, columns=4):
    slices = [int(x) for x in slices]
    rows = int(np.ceil(len(slices) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 4.2, rows * 4.0), squeeze=False)
    for n, index in enumerate(slices):
        ax = axes.ravel()[n]
        if plane == 'axial':
            image = data[:, :, index]
            xlabel, ylabel = 'j (x index)', 'i (y index)'
            aspect = 1.0
        elif plane == 'coronal':
            image = data[:, index, :]
            xlabel, ylabel = 'k (z index)', 'i (y index)'
            aspect = 'auto'
        elif plane == 'sagittal':
            image = data[index, :, :]
            xlabel, ylabel = 'k (z index)', 'j (x index)'
            aspect = 'auto'
        else:
            raise ValueError(plane)
        ax.imshow(hu_image(image), cmap='gray', vmin=-1000, vmax=1200,
                  origin='upper', aspect=aspect, interpolation='nearest')
        ax.set_title(f'{plane} index={index}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(slices):]:
        ax.axis('off')
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def orthogonal_triptych(data, axial, coronal, sagittal, out_path, title):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    panels = [
        ('axial', data[:, :, axial], 1.0, f'axial k={axial}', 'j', 'i'),
        ('coronal', data[:, coronal, :], 'auto', f'coronal j={coronal}', 'k', 'i'),
        ('sagittal', data[sagittal, :, :], 'auto', f'sagittal i={sagittal}', 'k', 'j'),
    ]
    for ax, (_, image, aspect, panel_title, xlabel, ylabel) in zip(axes, panels):
        ax.imshow(hu_image(image), cmap='gray', vmin=-1000, vmax=1200,
                  origin='upper', aspect=aspect, interpolation='nearest')
        ax.set_title(panel_title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ct', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    image, data, spacing = load_volume(args.ct)
    shape = list(data.shape)
    # The overview is deliberately independent of any existing segmentation.
    axial = list(range(0, shape[2], 4))
    if axial[-1] != shape[2] - 1:
        axial.append(shape[2] - 1)
    coronal = list(range(0, shape[1], 32))
    if coronal[-1] != shape[1] - 1:
        coronal.append(shape[1] - 1)
    sagittal = list(range(0, shape[0], 32))
    if sagittal[-1] != shape[0] - 1:
        sagittal.append(shape[0] - 1)
    montage(data, axial, 'axial', args.out / 'initial_axial_every4.png',
            'phase_10 source CT, axial overview; window -1000..1200 HU', columns=6)
    montage(data, coronal, 'coronal', args.out / 'initial_coronal_every32.png',
            'phase_10 source CT, coronal overview; window -1000..1200 HU', columns=5)
    montage(data, sagittal, 'sagittal', args.out / 'initial_sagittal_every32.png',
            'phase_10 source CT, sagittal overview; window -1000..1200 HU', columns=5)
    for k in [20, 30, 40, 50, 60, 70, 80]:
        if k < shape[2]:
            orthogonal_triptych(data, k, shape[1] // 2, shape[0] // 2,
                                args.out / f'initial_triptych_k{k:03d}.png',
                                f'phase_10 source CT overview at axial k={k}')
    metadata = {
        'source_ct': str(args.ct.resolve()),
        'shape_ijk': shape,
        'voxel_spacing_mm': [float(x) for x in spacing],
        'index_convention': 'a[i,j,k]; axial a[:,:,k], coronal a[:,j,:], sagittal a[i,:,:]',
        'display': {'origin': 'upper', 'window_hu': [-1000, 1200], 'interpolation': 'nearest'},
        'overview_indices': {'axial': axial, 'coronal': coronal, 'sagittal': sagittal},
        'source_header_units': image.header.get_xyzt_units(),
    }
    (args.out / 'initial_review_metadata.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
