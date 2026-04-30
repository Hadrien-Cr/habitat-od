import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image, ImageDraw
import cv2

def plot_semantic_2d_map(
    bg_grid,
    sem_grid,
    colors: dict[int, str],
    name_mapping: dict[int, str],
    scale=8
) -> Image.Image:
    # sem: (h, w, nc)
    num_sem_classes = sem_grid.shape[-1]

    colored = np.zeros((*sem_grid.shape[:2], 3), dtype=np.uint8)

    colored[bg_grid == 0] = [255, 255, 255]
    colored[bg_grid == 1] = [0, 0, 0]

    for c in range(num_sem_classes):
        object_mask = (sem_grid[:, :, c] == 1)
        colored[object_mask] = colors[c]

    # Convert to image and upscale x4
    img = Image.fromarray(colored)
    img = img.resize(
        (img.width * scale, img.height * scale),
        resample=Image.NEAREST  # type: ignore
    )

    draw = ImageDraw.Draw(img)

    for c in range(num_sem_classes):
        object_mask = (sem_grid[:, :, c] == 1)

        labeled_mask, num = ndimage.label(object_mask)

        for i in range(1, num + 1):
            region = labeled_mask == i
            coords = np.column_stack(np.where(region))

            if len(coords) < 30:
                continue

            y, x = coords.mean(axis=0)

            x *= scale
            y *= scale

            draw.text(
                (x, y),
                name_mapping[c],
                fill=(255,255,255),
                font_size=20,
                anchor="mm"  # center text on region centroid
            )
    return img

def plot_mask(mask) -> Image.Image:
    colored = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
    colored[mask] = [255, 255, 255]
    return Image.fromarray(colored)

def make_mosaic(
    list_fnames_images: list[tuple[str, np.ndarray]],
    target_height: int = 2000,
    N_cols: int = 4
) -> Image.Image:
    n =  len(list_fnames_images)
    processed_images = []

    for i, (filename, img) in enumerate(list_fnames_images):
        # add text overlay with filename
        cv2.putText(
            img,
            filename,
            (5, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        processed_images.append(img)


    # 3. Create the nx4 mosaic
    rows = []
    for i in range((n + N_cols - 1) // N_cols):
        # Stack 4 images horizontally (cols)
        start_idx = i * N_cols
        end_idx = (i + 1) * N_cols
        row_of_images = processed_images[start_idx:end_idx]
        if len(row_of_images) == 0:
            continue
        elif len(row_of_images) == N_cols:
            rows.append(np.hstack(row_of_images))
        else:
            # If not enough images to fill the last row, pad with black images
            n_missing = N_cols - len(row_of_images)
            black_image = np.zeros_like(processed_images[0])
            row_of_images.extend([black_image] * n_missing)
            rows.append(np.hstack(row_of_images))

    final_mosaic = np.vstack(rows)

    downscale_factor = target_height / final_mosaic.shape[0]
    final_mosaic = cv2.resize(
        final_mosaic,
        (
            int(final_mosaic.shape[1] * downscale_factor),
            int(final_mosaic.shape[0] * downscale_factor),
        ),
    )
    return Image.fromarray(final_mosaic)