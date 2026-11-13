import argparse
import csv
import os
import glob
import json
from pathlib import Path
import numpy as np
import openslide

from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

SVS_DIR = "data/images"

OUTPUT_DIR = "data/processed_patches"

PATCH_SIZE = 224

MAX_PATCHES_PER_SLIDE = 20

MIN_TISSUE_PERCENT = 50

THUMBNAIL_SIZE = 2000

EPSILON = 1e-8


# ============================================================
# FIND SVS FILES
# ============================================================

def find_svs_files():

    files = glob.glob(
        os.path.join(
            SVS_DIR,
            "**",
            "*.svs"
        ),
        recursive=True
    )

    return sorted(files)


# ============================================================
# TISSUE DETECTION
# ============================================================

def tissue_percentage(image):
    """
    Estimate how much of a patch contains tissue.

    White/near-white background is rejected.
    """

    image = np.asarray(
        image
    ).astype(
        np.float32
    )

    brightness = image.mean(
        axis=2
    )

    saturation = (
        image.max(axis=2)
        - image.min(axis=2)
    )

    tissue = (
        (brightness < 235)
        &
        (saturation > 10)
    )

    return (
        tissue.mean() * 100.0
    )


# ============================================================
# RGB -> OPTICAL DENSITY
# ============================================================

# Fixed H/E reference basis, columns ordered hematoxylin then eosin.
# Concentration targets are calibrated in this basis using local tissue patches.
REFERENCE_STAINS = np.array([[0.5626, 0.2159],
                             [0.7201, 0.8012],
                             [0.4062, 0.5581]], dtype=np.float64)
REFERENCE_STAINS /= np.linalg.norm(REFERENCE_STAINS, axis=0)
REFERENCE_PATH = Path(__file__).resolve().parents[1] / 'data' / 'metadata' / 'macenko_reference.json'


def load_reference_percentile(path=REFERENCE_PATH):
    """Load the local calibration; fail explicitly rather than use prototype values.

    calibrate_macenko.py selects eight spatially distributed patches per slide
    with seed 20260925 and >=70% tissue. P_ref is the componentwise median of
    patchwise 95th percentiles in the fixed reference reconstruction basis.
    Exact coordinates and measurements are recorded in the JSON.
    """
    with Path(path).open() as handle:
        config = json.load(handle)
    values = np.asarray(config['reference_percentile'], dtype=np.float64)
    matrix = np.asarray(config['reference_stain_matrix'], dtype=np.float64)
    if (config.get('schema_version') != 1 or config.get('percentile') != 95
            or config.get('stain_order') != ['hematoxylin', 'eosin']
            or values.shape != (2,) or not np.isfinite(values).all()
            or np.any(values <= 0) or matrix.shape != (3, 2)
            or not np.allclose(matrix, REFERENCE_STAINS, rtol=0, atol=1e-12)):
        raise ValueError('Invalid or incompatible Macenko reference calibration')
    return values

MAX_STAIN_CONDITION = 100.0
MIN_TISSUE_PIXELS = 100


def _finite(values, name):
    if not np.all(np.isfinite(values)):
        raise ValueError(name + " contains NaN or Inf")
    return values


def rgb_to_od(image):
    image = _finite(np.asarray(image, dtype=np.float64), "RGB")
    if np.any(image < 0) or np.any(image > 255):
        raise ValueError("RGB values must be in [0, 255]")
    return -np.log(np.maximum(image, 1.0) / 255.0)


def od_to_rgb(od):
    od = _finite(np.asarray(od, dtype=np.float64), "OD")
    return (255.0 * np.exp(-np.clip(od, 0.0, 10.0))).astype(np.uint8)


def _contract(left, right):
    # NumPy 2.0.2/macOS Accelerate matmul emits spurious FP warnings even
    # for bounded finite operands. Explicit contraction avoids that BLAS path;
    # optimize=False is intentional. No warning filters or invalid-value repair.
    _finite(left, "left matrix")
    _finite(right, "right matrix")
    return _finite(np.einsum('ij,jk->ik', left, right, optimize=False), "product")


def _valid_stains(matrix):
    if matrix.shape != (3, 2) or not np.all(np.isfinite(matrix)):
        return False
    if np.any(matrix < 0):
        return False
    try:
        condition = np.linalg.cond(matrix)
        return np.isfinite(condition) and condition <= MAX_STAIN_CONDITION
    except np.linalg.LinAlgError:
        return False


def _estimate_stains(tissue):
    """Standard Macenko plane/angular estimation, or fixed reference fallback."""
    fallback = REFERENCE_STAINS.copy()
    if not _valid_stains(fallback):
        raise ValueError("Invalid reference stain matrix")
    try:
        _, singular, vt = np.linalg.svd(tissue - tissue.mean(axis=0),
                                       full_matrices=False)
        if (not np.all(np.isfinite(singular)) or not np.all(np.isfinite(vt))
                or singular[0] <= EPSILON
                or singular[1] <= singular[0] / MAX_STAIN_CONDITION):
            return fallback
        plane = vt[:2].T.copy()
        # Orient the first axis into the positive OD cone to avoid angle wrapping.
        if np.dot(plane[:, 0], tissue.mean(axis=0)) < 0:
            plane[:, 0] *= -1
        # Centering is for covariance/SVD only; stain rays start at OD zero.
        projected = _contract(tissue, plane)
        angles = np.arctan2(projected[:, 1], projected[:, 0])
        lo, hi = np.percentile(angles, [1, 99])
        directions = np.array([[np.cos(lo), np.cos(hi)],
                               [np.sin(lo), np.sin(hi)]])
        matrix = _contract(plane, directions)
        matrix /= np.linalg.norm(matrix, axis=0)
        # Standard H/E ordering by red OD component.
        if matrix[0, 0] < matrix[0, 1]:
            matrix = matrix[:, ::-1]
        return matrix if _valid_stains(matrix) else fallback
    except np.linalg.LinAlgError:
        return fallback


def modified_macenko(image):
    """Normalize one RGB patch; leave insufficient tissue unchanged.

    Standard components: OD, SVD stain plane, angular percentiles, least-squares
    concentrations, percentile scaling, reconstruction in reference stain space.
    Proposed experimental component: adaptive alpha in
        C' = C * (P_ref / (P_source + epsilon)) ** alpha.
    Uses calibrated local reference concentrations and the 95th percentile.
    Alpha = clip(1 / (1 + mean(std(C, axis=1))), 0.5, 1.0), computed per patch.
    The adaptive equation is our proposed modification, not standard Macenko.
    Invalid RGB input raises ValueError instead of silently replacing values.
    """
    values = np.asarray(image, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 3 or values.size == 0:
        raise ValueError("Expected a nonempty H x W x 3 RGB patch")
    od = rgb_to_od(values).reshape(-1, 3)
    original = values.astype(np.uint8)
    # Standard OD background exclusion plus the pipeline's chromatic tissue gate.
    flat = values.reshape(-1, 3)
    mask = ((od > 0.15).all(axis=1) & (np.ptp(flat, axis=1) > 10))
    if np.count_nonzero(mask) < MIN_TISSUE_PIXELS:
        return original.copy()
    tissue = od[mask]
    stains = _estimate_stains(tissue)
    try:
        concentrations = np.linalg.lstsq(stains, tissue.T, rcond=None)[0]
    except np.linalg.LinAlgError:
        concentrations = np.linalg.lstsq(REFERENCE_STAINS, tissue.T, rcond=None)[0]
    concentrations = np.maximum(_finite(concentrations, "concentrations"), 0.0)
    source_percentile = np.percentile(concentrations, 95, axis=1)
    # No meaningful two-stain scaling when either stain is absent.
    if np.any(source_percentile <= EPSILON):
        return original.copy()
    reference_percentile = load_reference_percentile()

    # PROPOSED MODIFICATION: preserve the equation without clipping its ratio.
    variation = np.mean(np.std(concentrations, axis=1))
    alpha = np.clip(1.0 / (1.0 + variation), 0.5, 1.0)
    scaling = (reference_percentile / (source_percentile + EPSILON)) ** alpha
    normalized_concentrations = _finite(concentrations * scaling[:, None],
                                        "normalized concentrations")
    normalized_od = _contract(REFERENCE_STAINS, normalized_concentrations)
    result = original.copy().reshape(-1, 3)
    result[mask] = od_to_rgb(normalized_od.T)
    return result.reshape(original.shape)


# ============================================================
# PROCESS ONE SVS SLIDE
# ============================================================

def process_slide(svs_path):

    print(
        "\nProcessing:",
        svs_path
    )

    # --------------------------------------------------------
    # Open whole-slide image
    # --------------------------------------------------------

    try:

        slide = openslide.OpenSlide(
            svs_path
        )

    except Exception as e:

        print(
            "Could not open slide:",
            e
        )

        return

    slide_width, slide_height = (
        slide.dimensions
    )

    print(
        "Slide dimensions:",
        slide.dimensions
    )

    # --------------------------------------------------------
    # Thumbnail for tissue localization
    # --------------------------------------------------------

    thumbnail = slide.get_thumbnail(
        (
            THUMBNAIL_SIZE,
            THUMBNAIL_SIZE
        )
    )

    thumbnail = thumbnail.convert(
        "RGB"
    )

    thumb_width, thumb_height = (
        thumbnail.size
    )

    scale_x = (
        slide_width /
        thumb_width
    )

    scale_y = (
        slide_height /
        thumb_height
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    slide_name = os.path.splitext(
        os.path.basename(
            svs_path
        )
    )[0]

    output_folder = os.path.join(
        OUTPUT_DIR,
        slide_name
    )

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    patch_count = 0

    # --------------------------------------------------------
    # Scan thumbnail
    # --------------------------------------------------------

    for y in range(
        0,
        thumb_height - PATCH_SIZE,
        PATCH_SIZE
    ):

        if patch_count >= MAX_PATCHES_PER_SLIDE:
            break

        for x in range(
            0,
            thumb_width - PATCH_SIZE,
            PATCH_SIZE
        ):

            if patch_count >= MAX_PATCHES_PER_SLIDE:
                break

            # ----------------------------------------------
            # Thumbnail patch
            # ----------------------------------------------

            thumb_patch = thumbnail.crop(
                (
                    x,
                    y,
                    x + PATCH_SIZE,
                    y + PATCH_SIZE
                )
            )

            tissue_pct = tissue_percentage(
                thumb_patch
            )

            if (
                tissue_pct
                <
                MIN_TISSUE_PERCENT
            ):
                continue

            # ----------------------------------------------
            # Convert thumbnail coordinates
            # to original SVS coordinates
            # ----------------------------------------------

            svs_x = int(
                x * scale_x
            )

            svs_y = int(
                y * scale_y
            )

            # Read a genuinely small level-0 patch, not a thumbnail-sized
            # field expanded into a huge region and then downsampled.
            svs_width = PATCH_SIZE
            svs_height = PATCH_SIZE
            svs_x = min(svs_x, slide_width - PATCH_SIZE)
            svs_y = min(svs_y, slide_height - PATCH_SIZE)

            # ----------------------------------------------
            # Read only this region
            # ----------------------------------------------

            try:

                patch = slide.read_region(
                    (
                        svs_x,
                        svs_y
                    ),
                    0,
                    (
                        svs_width,
                        svs_height
                    )
                ).convert(
                    "RGB"
                )

            except Exception as e:

                print(
                    "Patch extraction error:",
                    e
                )

                continue

            if tissue_percentage(patch) < MIN_TISSUE_PERCENT:
                continue

            # ----------------------------------------------
            # Modified Macenko
            # ----------------------------------------------

            patch_array = np.array(
                patch
            )

            normalized_patch = (
                modified_macenko(
                    patch_array
                )
            )

            normalized_patch = Image.fromarray(
                normalized_patch
            )

            # ----------------------------------------------
            # Save processed patch
            # ----------------------------------------------

            output_path = os.path.join(
                output_folder,
                f"patch_{patch_count:04d}.png"
            )

            normalized_patch.save(
                output_path
            )

            patch_count += 1

    slide.close()

    print(
        "Patches extracted:",
        patch_count
    )


# ============================================================
# MAIN
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "preprocessing_pilot"
TARGET_MPP = 0.5


def _read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def cohort_slides():
    """Return only selected, validated slides with consistent cohort metadata."""
    patients = {row["case_uuid"]: row for row in _read_csv(
        ROOT / "data/metadata/patients_250.csv")}
    selected = _read_csv(ROOT / "data/metadata/selected_500_slides.csv")
    status = {row["slide_file_uuid"]: row for row in _read_csv(
        ROOT / "data/metadata/download_status_500.csv")}
    mapping = {row["slide_file_uuid"]: row for row in _read_csv(
        ROOT / "data/metadata/patient_slide_mapping.csv")}
    available = []
    for row in selected:
        patient = patients.get(row["case_uuid"])
        mapped = mapping.get(row["slide_file_uuid"])
        if (patient is None or mapped is None
                or patient["case_id"] != row["patient_id"]
                or mapped["case_uuid"] != row["case_uuid"]
                or status.get(row["slide_file_uuid"], {}).get("validation_result") != "passed"):
            continue
        path = ROOT / "data/images/tcga_brca_500" / row["patient_id"] / row["slide_file_name"]
        if path.is_file():
            available.append((row, patient, path))
    return available


def _balanced_pilot(slides, limit):
    """Select patients round-robin by the image-baseline diagnosis label."""
    groups = {"ductal": [], "other": []}
    for item in slides:
        row, patient, _ = item
        diagnosis = patient["primary_diagnosis"].strip().lower()
        if diagnosis in {"", "not reported", "unknown", "not available"}:
            continue
        label = "ductal" if "duct" in diagnosis else "other"
        if patient["case_uuid"] not in {p[1]["case_uuid"] for p in groups[label]}:
            groups[label].append(item)
    chosen = []
    for i in range(limit):
        group = groups["ductal"] if i % 2 == 0 else groups["other"]
        index = i // 2
        if index < len(group):
            chosen.append(group[index])
    return chosen


def process_cohort_slide(item, output_dir, max_patches):
    row, patient, path = item
    output_dir = Path(output_dir).resolve()
    records = []
    with openslide.OpenSlide(str(path)) as slide:
        width, height = slide.dimensions
        thumb = slide.get_thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE)).convert("RGB")
        tw, th = thumb.size
        mpp = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
        if mpp:
            desired_downsample = TARGET_MPP / float(mpp)
            level = min(range(slide.level_count),
                        key=lambda i: abs(slide.level_downsamples[i] - desired_downsample))
        else:
            level = 0
        downsample = float(slide.level_downsamples[level])
        # Scan thumbnail cells matching the physical footprint of a 224px patch.
        cell_w = max(1, int(round(PATCH_SIZE * downsample * tw / width)))
        cell_h = max(1, int(round(PATCH_SIZE * downsample * th / height)))
        out_dir = Path(output_dir) / row["slide_file_uuid"]
        out_dir.mkdir(parents=True, exist_ok=True)
        for ty in range(0, max(1, th - cell_h + 1), cell_h):
            for tx in range(0, max(1, tw - cell_w + 1), cell_w):
                if len(records) >= max_patches:
                    break
                crop = thumb.crop((tx, ty, min(tw, tx + cell_w), min(th, ty + cell_h)))
                if tissue_percentage(crop) < MIN_TISSUE_PERCENT:
                    continue
                x = min(int(tx * width / tw), max(0, width - int(PATCH_SIZE * downsample)))
                y = min(int(ty * height / th), max(0, height - int(PATCH_SIZE * downsample)))
                try:
                    patch = slide.read_region((x, y), level, (PATCH_SIZE, PATCH_SIZE)).convert("RGB")
                    tissue_pct = tissue_percentage(patch)
                    if tissue_pct < MIN_TISSUE_PERCENT:
                        continue
                    normalized = modified_macenko(np.asarray(patch))
                    destination = out_dir / f"patch_{len(records):04d}.png"
                    Image.fromarray(normalized).save(destination)
                    try:
                        patch_path = destination.relative_to(ROOT)
                    except ValueError:
                        # Absolute paths are accepted by downstream manifest readers.
                        patch_path = destination
                    records.append({
                        "patient_id": row["patient_id"], "case_uuid": row["case_uuid"],
                        "slide_file_uuid": row["slide_file_uuid"],
                        "patch_file": str(patch_path),
                        "x": x, "y": y, "level": level,
                        "mpp_x": mpp or "", "tissue_percent": "%.6f" % tissue_pct,
                        "normalization_method": "modified_macenko",
                    })
                except Exception as error:
                    print("Skipping patch", row["slide_file_uuid"], x, y, error)
            if len(records) >= max_patches:
                break
    return records


def main():
    parser = argparse.ArgumentParser(description="Extract normalized patches from validated cohort slides")
    parser.add_argument("--limit-patients", type=int, default=6,
                        help="Balanced pilot size; use 0 to process all available labeled patients")
    parser.add_argument("--max-patches-per-slide", type=int, default=4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.limit_patients < 0 or args.max_patches_per_slide < 1:
        parser.error("limits must be nonnegative and max patches must be positive")
    slides = cohort_slides()
    chosen = slides if args.limit_patients == 0 else _balanced_pilot(slides, args.limit_patients)
    if not chosen:
        parser.exit(1, "No eligible validated cohort slides found.\n")
    manifest = []
    for item in chosen:
        records = process_cohort_slide(item, args.output, args.max_patches_per_slide)
        manifest.extend(records)
        print(item[0]["patient_id"], len(records), "patches")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "model_patches.csv"
    columns = ["patient_id", "case_uuid", "slide_file_uuid", "patch_file", "x", "y",
               "level", "mpp_x", "tissue_percent", "normalization_method"]
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(manifest)
    print("Saved", len(manifest), "patches to", manifest_path)


if __name__ == "__main__":
    main()
