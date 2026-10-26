
import json
import numpy as np
import cv2
from PIL import Image
import torch
from tqdm import tqdm
import pickle
from pathlib import Path
from skimage import color
from PIL import ImageCms
from segment_anything import sam_model_registry, SamPredictor

np.random.seed(0)

XGBOOST_THRESHOLD = 0.9998

# hsv filter for senescent pixels
def hsv_filter_senescent(image_bgr, mask, s_thresh=50, v_thresh=160):
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(image_hsv)
    dark_pixels = ((s < s_thresh) | (v < v_thresh))
    mask_clean = mask.copy()
    mask_clean[(mask == 2) & dark_pixels] = 0
    return mask_clean

# resize image to max dimension
def resize_max_dim(img, max_dim=1024):
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
    return img, scale

# compute excess green index
def excess_green(rgb):
    r = rgb[...,0].astype(np.float32)
    g = rgb[...,1].astype(np.float32)
    b = rgb[...,2].astype(np.float32)
    exg = 2*g - r - b
    exg -= exg.min()
    denom = exg.max() if exg.max() > 0 else 1.0
    return exg / denom

# sample points spread across mask
def sample_points_spread(mask, k):
    H, W = mask.shape
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.empty((0,2), dtype=np.int32)
    pts = []
    grid = int(np.sqrt(k)) + 1
    for gy in range(grid):
        for gx in range(grid):
            y0 = int(gy * H / grid); y1 = int((gy+1) * H / grid)
            x0 = int(gx * W / grid); x1 = int((gx+1) * W / grid)
            in_cell = (ys >= y0) & (ys < y1) & (xs >= x0) & (xs < x1)
            idx = np.where(in_cell)[0]
            if len(idx) > 0:
                j = np.random.choice(idx)
                pts.append([xs[j], ys[j]])
            if len(pts) >= k:
                break
        if len(pts) >= k:
            break
    if len(pts) < k:
        remaining = min(k - len(pts), len(xs))
        rand_idx = np.random.choice(len(xs), size=remaining, replace=False)
        for j in rand_idx:
            pts.append([xs[j], ys[j]])
    return np.array(pts, dtype=np.int32)

# choose best mask from candidates
def choose_best_mask(masks, pos_pts, neg_pts):
    best_idx, best_score = 0, -1e9
    H, W = masks[0].shape
    for i, m in enumerate(masks):
        m_bool = m.astype(bool)
        pos_hit = int(m_bool[pos_pts[:,1], pos_pts[:,0]].sum()) if len(pos_pts) else 0
        neg_hit = int(m_bool[neg_pts[:,1], neg_pts[:,0]].sum()) if len(neg_pts) else 0
        area_ratio = m_bool.mean()
        score = pos_hit - 2*neg_hit
        if area_ratio > 0.9:
            score -= 10
        if score > best_score:
            best_score, best_idx = score, i
    return best_idx

# postprocess mask to remove small components
def postprocess_mask(mask, min_area=500):
    mask = mask.astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    num_labels, labels = cv2.connectedComponents(mask)
    if num_labels <= 1:
        return mask
    areas = np.bincount(labels.ravel())
    keep = np.where(areas >= min_area)[0]
    keep[keep == 0] = -1
    keep_mask = np.isin(labels, keep)
    return keep_mask.astype(np.uint8)

# generate plant mask using SAM
def generate_plant_mask(image_rgb, sam_ckpt, min_component=500, max_dim=1024):
    small, scale = resize_max_dim(image_rgb, max_dim)
    exg = excess_green(small)
    q60 = np.quantile(exg, 0.95)  # higher = stricter, e.g. 0.90–0.98
    q25 = np.quantile(exg, 0.20)  # lower = stricter, e.g. 0.10–0.30
    pos_cand = exg > q60
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    S = hsv[...,1].astype(np.float32) / 255.0
    B = small[...,2].astype(np.float32)
    neg_cand = (exg < q25) | (S < 0.18) | (B > np.quantile(B, 0.70))  # less strict
    # sample more points for better coverage
    pos_pts = sample_points_spread(pos_cand, 80)
    neg_pts = sample_points_spread(neg_cand, 24)
    # add a few corner negatives to discourage global fill
    Hs, Ws = small.shape[:2]
    corner_pad = 8
    corner_pts = np.array([[corner_pad, corner_pad], [Ws - corner_pad, corner_pad], [corner_pad, Hs - corner_pad], [Ws - corner_pad, Hs - corner_pad]], dtype=np.int32)
    neg_pts = np.vstack([neg_pts, corner_pts])
    # run SAM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_b"](checkpoint=sam_ckpt)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(small)
    pts = np.vstack([pos_pts, neg_pts]).astype(np.float32)
    labels = np.concatenate([np.ones(len(pos_pts), dtype=np.int32), np.zeros(len(neg_pts), dtype=np.int32)])
    masks, scores, logits = predictor.predict(point_coords=pts, point_labels=labels, multimask_output=True)
    # pick best mask
    best = choose_best_mask(masks, pos_pts, neg_pts)
    mask_small = masks[best].astype(np.uint8)
    # postprocess
    mask_small = postprocess_mask(mask_small, min_component)
    # upscale
    H, W = image_rgb.shape[:2]
    if scale != 1.0:
        plant_mask = cv2.resize(mask_small, (W, H), interpolation=cv2.INTER_NEAREST)
    else:
        plant_mask = mask_small
    return plant_mask

# get features from rectangle_fields.py
def get_features(image):
    pil_image = Image.fromarray(image)

    hsv = np.array(pil_image.convert(mode='HSV'))
    srgb_p = ImageCms.createProfile("sRGB")
    lab_p  = ImageCms.createProfile("LAB")
    rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
    Lab = np.array(ImageCms.applyTransform(pil_image, rgb2lab))
    ycbcr = np.array(pil_image.convert(mode='YCbCr'))
    Labb = color.rgb2lab(image)
    r = image[:, :,0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    h = (hsv[:, :, 0].astype(np.float32) * 360) / 255
    s = (hsv[:, :, 1]) / 2.55
    a = Labb[:, :, 1]
    bb = Lab[:, :, 2]
    ge =  np.mean([r,g,b], axis = 0)

    CMYlist = [1 - r / 255, 1 - g / 255, 1 - b / 255]
    CMYlist = np.array([np.min(idx) for idx in zip(*CMYlist)])
    m = ((1 - g / 255 - CMYlist ) / (1 - CMYlist )) * 100
    ye = ((1 - b / 255 - CMYlist ) / (1 - CMYlist )) * 100

    cb = ycbcr[:, :, 1]
    cr = ycbcr[:, :, 2]

    i = (0.596 * r - 0.275 * g - 0.321 * b)
    q = (0.212 * r - 0.523 * g + 0.311 * b)

    model_input = np.stack((r,g,b,h,s,a,bb,ge,m,ye,cb,cr,i,q), axis=2).squeeze(0)
    model_input = np.nan_to_num(model_input) # handle black pixels (avoiding Input contains NaN, infinity or a value too large for dtype('float64') error)

    return model_input

# use the same prediction_XG_SVM as in rectangle_fields.py (with get_features)
def prediction_XG_SVM(image, model, threshold, contrasted, mask):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    if mask is None:
        mask = np.ones((height, width))
    _validate_mask(mask, (height, width), "plant_mask")
    if not np.any(mask > 0):
        return np.zeros((height, width), dtype=np.uint8), np.zeros((height, width), dtype=np.float32)
    image = image.reshape((width*height), 1, 3)
    mask = mask.reshape((width*height), 1)
    yellow_green_mask = np.zeros(image.shape[:-1])
    vegetation_pixels = mask > 0
    image = image[None, vegetation_pixels]
    featured_image = get_features(image)
    probas = model.predict_proba(featured_image)[:,1]
    y_pred = (probas >= threshold).astype(int)
    yellow_green_mask[vegetation_pixels] = y_pred
    mask[(mask == 1) & (yellow_green_mask != 1)] = 2
    mask = mask.reshape((height, width))
    prob_map = np.zeros_like(yellow_green_mask, dtype=np.float32)
    prob_map[vegetation_pixels] = probas
    prob_map = prob_map.reshape((height, width))
    return mask, prob_map

# visualize necrosis and yellow leaf detection
def visualisation_Necrosis(rgb_image: np.ndarray, yg_mask: np.ndarray, stg) -> np.ndarray:
    """
    Legacy visualization: both input and returned image use OpenCV BGR.
    """
    image_copy = rgb_image.copy()

    # works because BGR
    if stg == "SegVeg_Necrosis":
        image_copy[yg_mask == 0] = (0, 0, 0)         # background = black
        image_copy[yg_mask == 1] = (34, 70, 34)      # green = dark dull green (BGR)
        image_copy[yg_mask == 3] = (0, 255, 255)     # chlorosis = bright yellow (BGR)
        image_copy[yg_mask == 2] = (0, 0, 255)       # necrosis = bright red
    if stg == "SegVeg_XG_SVM":
        image_copy[yg_mask == 0] = (0, 0, 0)         # background = black
        image_copy[yg_mask == 1] = (34, 70, 34)      # green = dark dull green (BGR)
        image_copy[yg_mask == 2] = (0, 255, 255)     # unhealthy (senescent) = bright yellow (BGR)

    visualisation = cv2.addWeighted(rgb_image, 0.4, image_copy, 0.6, 0)

    index_class = np.unique(yg_mask)
    color_vizu = []
    # update legend colors: 0=black, 1=dark dull green, 2=red, 3=bright yellow
    id = [(0, 0, 0), (34, 70, 34), (0, 0, 255), (0, 255, 255)]
    for c in index_class:
        color_vizu.append(id[int(c)])

    return visualisation, index_class, color_vizu

# split big image into tiles with overlap
def tile_image(img, tile_size=1024, overlap=64):
    _validate_rgb_image(img)
    if not isinstance(tile_size, int) or tile_size <= 0:
        raise ValueError("tile_size must be a positive integer")
    if not isinstance(overlap, int) or not 0 <= overlap < tile_size:
        raise ValueError("overlap must be an integer with 0 <= overlap < tile_size")
    h, w = img.shape[:2]
    tiles = []
    for y in range(0, h, tile_size - overlap):
        for x in range(0, w, tile_size - overlap):
            y1, y2 = y, min(y + tile_size, h)
            x1, x2 = x, min(x + tile_size, w)
            tile = img[y1:y2, x1:x2]
            tiles.append(((y1, y2, x1, x2), tile))
    return tiles, (h, w)

# stitch tiled predictions back into full-size array
def stitch_tiles(tiles, shape):
    if len(shape) != 2 or any(not isinstance(d, (int, np.integer)) or d <= 0 for d in shape):
        raise ValueError(f"Invalid stitching shape: {shape}; expected positive (height, width)")
    H, W = shape
    out = np.zeros((H, W), dtype=np.uint8)
    for (y1, y2, x1, x2), tile_mask in tiles:
        if not (0 <= y1 < y2 <= H and 0 <= x1 < x2 <= W):
            raise ValueError(f"Invalid tile coordinates {(y1, y2, x1, x2)} for shape {shape}")
        _validate_mask(tile_mask, (y2 - y1, x2 - x1), "tile_mask")
        out[y1:y2, x1:x2] = np.maximum(out[y1:y2, x1:x2], tile_mask)
    return out


def _validate_rgb_image(image_rgb):
    if (not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3
            or image_rgb.shape[2] != 3 or image_rgb.size == 0
            or image_rgb.dtype != np.uint8):
        raise ValueError("image_rgb must be a nonempty uint8 RGB image with shape (height, width, 3)")


def _validate_mask(mask, shape, name):
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.shape != tuple(shape) or mask.size == 0:
        raise ValueError(f"{name} must be a nonempty 2D mask with shape {tuple(shape)}; "
                         f"got {getattr(mask, 'shape', None)}")


def summarize_masks(plant_mask, final_mask, image_name):
    """Count stitched SAM vegetation separately from final post-HSV classes."""
    if not isinstance(final_mask, np.ndarray) or final_mask.ndim != 2 or final_mask.size == 0:
        raise ValueError("final_mask must be a nonempty 2D classification mask")
    _validate_mask(plant_mask, final_mask.shape, "plant_mask")
    if not np.all(np.isin(plant_mask, [0, 1, 255])):
        raise ValueError("plant_mask must contain only binary values 0/1 or 0/255")
    if not np.all(np.isin(final_mask, [0, 1, 2])):
        raise ValueError("final_mask must contain only classes 0 (other), 1 (green), 2 (senescent)")
    height, width = final_mask.shape
    total = int(final_mask.size)
    sam_pixels = int(np.count_nonzero(plant_mask))
    green = int(np.count_nonzero(final_mask == 1))
    yellow = int(np.count_nonzero(final_mask == 2))
    classified = green + yellow
    return {
        "image_name": Path(image_name).name,
        "image_width": width,
        "image_height": height,
        "total_pixels": total,
        "sam_vegetation_pixels": sam_pixels,
        "green_pixels": green,
        "yellow_pixels": yellow,
        "other_pixels": total - classified,
        "valid_classified_vegetation_pixels": classified,
        "vegetation_coverage_percent": 100.0 * sam_pixels / total,
        "yellow_ratio_percent": 100.0 * yellow / classified if classified else 0.0,
        "metric_definitions": {
            "sam_vegetation_pixels": "Nonzero pixels in the stitched generate_plant_mask output, copied before SegVeg classification; includes existing SAM mask postprocessing.",
            "green_pixels": "Pixels with final stitched class 1 after the existing HSV filter.",
            "yellow_pixels": "Pixels with final stitched class 2 (senescent vegetation) after the existing HSV filter.",
            "other_pixels": "total_pixels - green_pixels - yellow_pixels; includes background and vegetation removed by the HSV filter.",
            "vegetation_coverage_percent": "100 * sam_vegetation_pixels / total_pixels.",
            "yellow_ratio_percent": "100 * yellow_pixels / (green_pixels + yellow_pixels); 0.0 when the denominator is zero (no valid classified vegetation).",
        },
        "stitching_policy": "Existing np.maximum: SAM masks merge by union; final class 2 overrides class 1 in overlaps, which can bias yellow statistics upward.",
        "hsv_filter_behavior": "Preserved existing call: RGB tile passed to hsv_filter_senescent, whose conversion is COLOR_BGR2HSV; thresholds unchanged. The current filter uses only S/V, which are invariant to swapping red and blue.",
        "interpretation": "Model output pixel statistics only; not measured disease incidence or true growth abnormality rate. No physical area is estimated.",
    }


def create_abnormal_overlay(image_rgb, yellow_mask, alpha=0.5):
    """Blend RGB red only onto selected pixels; leave every other pixel exact."""
    _validate_rgb_image(image_rgb)
    _validate_mask(yellow_mask, image_rgb.shape[:2], "yellow_mask")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    selected = yellow_mask > 0
    overlay = image_rgb.copy()
    overlay[selected] = np.rint(
        (1 - alpha) * image_rgb[selected] + alpha * np.array([255, 0, 0])
    ).astype(np.uint8)
    return overlay


def save_diagnostic_outputs(image_rgb, plant_mask, final_mask, image_name, out_dir):
    """Export full-resolution diagnostics, the legacy visualization and JSON."""
    _validate_rgb_image(image_rgb)
    _validate_mask(final_mask, image_rgb.shape[:2], "final_mask")
    stats = summarize_masks(plant_mask, final_mask, image_name)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    yellow_mask = (final_mask == 2).astype(np.uint8) * 255
    Image.fromarray((plant_mask > 0).astype(np.uint8) * 255).save(out_dir / "plant_mask.png")
    Image.fromarray(yellow_mask).save(out_dir / "yellow_mask.png")
    Image.fromarray(create_abnormal_overlay(image_rgb, yellow_mask)).save(out_dir / "abnormal_overlay.png")

    # The legacy palette and addWeighted operate in BGR; Pillow saves RGB.
    visu, _, _ = visualisation_Necrosis(
        cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), final_mask, "SegVeg_XG_SVM")
    Image.fromarray(cv2.cvtColor(visu, cv2.COLOR_BGR2RGB)).save(out_dir / "Disease Detection.png")
    with (out_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(stats, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")

    print(f"\n模型识别结果统计: {stats['image_name']}")
    print(f"影像尺寸: {stats['image_width']} × {stats['image_height']}; 总像素: {stats['total_pixels']:,}")
    print(f"SAM 植被像素（分类前独立掩膜）: {stats['sam_vegetation_pixels']:,}")
    print(f"最终绿色像素（类别 1）: {stats['green_pixels']:,}")
    print(f"最终黄化/衰老像素（类别 2，HSV 后）: {stats['yellow_pixels']:,}")
    print(f"其他像素（含 HSV 剔除像素）: {stats['other_pixels']:,}")
    print(f"SAM 植被覆盖比例（SAM 植被 / 总像素）: {stats['vegetation_coverage_percent']:.4f}%")
    print(f"黄化比例（黄化 / 最终绿色与黄化像素之和 = {stats['valid_classified_vegetation_pixels']:,}）: "
          f"{stats['yellow_ratio_percent']:.4f}%")
    if not stats["valid_classified_vegetation_pixels"]:
        print("最终有效分类植被像素为 0，黄化比例按 0.0% 输出。")
    print("统计保留 np.maximum 拼接规则：重叠区类别 2 优先于类别 1，可能增加黄化计数。")
    print("以上仅为模型输出像素统计，不代表真实病害发生率或长势异常率；未估算实际面积。")
    print(f"Saved diagnostic outputs to {out_dir.resolve()}")
    return stats


# main segmentation pipeline for orthomosaic
def segveg_on_orthomosaic(orthomosaic_path, sam_ckpt, model_XG, out_dir, tile_size=1024, overlap=64):
    if not Path(orthomosaic_path).is_file():
        raise FileNotFoundError(f"Input image not found: {Path(orthomosaic_path).resolve()}")
    orig_img = cv2.imread(str(orthomosaic_path), cv2.IMREAD_COLOR)
    if orig_img is None or orig_img.size == 0:
        raise ValueError(f"Unable to read input image or image is empty: {Path(orthomosaic_path).resolve()}")
    if not Path(sam_ckpt).is_file():
        raise FileNotFoundError(f"SAM checkpoint not found: {Path(sam_ckpt).resolve()}")
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    # step 1: tile orthomosaic
    tiles, shape = tile_image(orig_img, tile_size=tile_size, overlap=overlap)

    stitched_masks = []
    stitched_plant_masks = []

    for (y1, y2, x1, x2), tile in tqdm(tiles, desc="Processing tiles"):
        # plant mask
        plant_mask = generate_plant_mask(tile, sam_ckpt)
        _validate_mask(plant_mask, tile.shape[:2], "plant_mask")
        # prediction_XG_SVM mutates mask through a reshape view. Preserve SAM first.
        stitched_plant_masks.append(((y1, y2, x1, x2), plant_mask.copy()))

        # SegVeg classification
        yg_mask, _ = prediction_XG_SVM(
            cv2.cvtColor(tile, cv2.COLOR_RGB2BGR),
            model_XG,
            threshold=0.8,
            contrasted=1,
            mask=plant_mask
        )
        # Preserve existing RGB input to this BGR-based filter for identical labels.
        yg_mask = hsv_filter_senescent(tile, yg_mask)

        stitched_masks.append(((y1, y2, x1, x2), yg_mask))

    # step 2: stitch results back
    full_mask = stitch_tiles(stitched_masks, shape)
    full_plant_mask = stitch_tiles(stitched_plant_masks, shape)

    return save_diagnostic_outputs(orig_img, full_plant_mask, full_mask, orthomosaic_path, out_dir)


def main():
    # Keep the original pickle loading sequence and XGBoost compatibility attributes.
    model_paths = ["SegVeg_data/model_scikit", "SegVeg_data/XGBoost", "SegVeg_data/Necrosis.pkl"]
    for model_path in model_paths:
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Model file not found: {Path(model_path).resolve()}")
    model_path = "SegVeg_data/model_scikit"
    with Path(model_path).open("rb") as stream:
        model_SVM = pickle.load(stream)

    model_path = "SegVeg_data/XGBoost"
    with Path(model_path).open("rb") as stream:
        model_XG = pickle.load(stream)
    new_attrs = ['grow_policy', 'max_bin', 'eval_metric', 'callbacks', 'early_stopping_rounds', 'max_cat_to_onehot', 'max_leaves', 'sampling_method', 'enable_categorical', 'feature_types', 'max_cat_threshold', 'predictor']
    for attr in new_attrs:
        setattr(model_XG, attr, None)

    model_path = "SegVeg_data/Necrosis.pkl"
    with Path(model_path).open("rb") as stream:
        model_Necrosis = pickle.load(stream)

    segveg_on_orthomosaic(
        orthomosaic_path="data/2025-08-27/Full Orthomosaic (10 meter).png",
        sam_ckpt="code/yellow_leaf_detection/weights/sam_vit_b.pth",
        model_XG=model_XG,
        out_dir="results/",
        tile_size=2048,
        overlap=256
    )


if __name__ == "__main__":
    main()
