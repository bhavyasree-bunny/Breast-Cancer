"""Calibrate a small local reference; never download slides.

Selection: split row-ordered thumbnail tissue candidates into eight spatial
bands, shuffle each with seed 20260925, and accept one 224x224 level-0 patch
per band. Require >=70% tissue and >=100 normalization-mask pixels. Try at
most eight candidates per band. Equal patch counts give slides equal weight.
P_ref: componentwise median of patchwise 95th-percentile nonnegative
least-squares H/E concentrations in the fixed reconstruction reference basis.
This is a local pilot reference, not a population-validated TCGA reference.
"""
import json
from pathlib import Path
import warnings

import numpy as np
import openslide

from image_preprocessing import (REFERENCE_STAINS, EPSILON, MIN_TISSUE_PIXELS,
                                 rgb_to_od, tissue_percentage)

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / 'data' / 'metadata' / 'macenko_reference.json'
SEED = 20260925
PATCHES_PER_SLIDE = 8


def main():
    rng = np.random.default_rng(SEED)
    slides = sorted((ROOT / 'data' / 'images').rglob('*.svs'))
    if not slides or len(slides) > 3:
        raise ValueError('Pilot calibration requires one to three existing slides')
    records = []
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        for source in slides:
            with openslide.OpenSlide(str(source)) as slide:
                thumb = np.asarray(slide.get_thumbnail((512, 512)).convert('RGB'))
                ys, xs = np.where((thumb.mean(axis=2) < 210)
                                  & (np.ptp(thumb, axis=2) > 30))
                bands = np.array_split(np.arange(len(xs)), PATCHES_PER_SLIDE)
                accepted = []
                for band_number, band in enumerate(bands):
                    for index in rng.permutation(band)[:8]:
                        x = min(int(xs[index] * slide.dimensions[0] / thumb.shape[1]),
                                slide.dimensions[0] - 224)
                        y = min(int(ys[index] * slide.dimensions[1] / thumb.shape[0]),
                                slide.dimensions[1] - 224)
                        if any(abs(x - px) < 224 and abs(y - py) < 224
                               for px, py in accepted):
                            continue
                        patch = np.asarray(slide.read_region((x, y), 0, (224, 224)).convert('RGB'))
                        percent = tissue_percentage(patch)
                        if percent < 70:
                            continue
                        od = rgb_to_od(patch).reshape(-1, 3)
                        mask = (od > 0.15).all(axis=1) & (np.ptp(patch.reshape(-1, 3), axis=1) > 10)
                        if mask.sum() < MIN_TISSUE_PIXELS:
                            continue
                        # Common reference basis: these concentrations describe
                        # the same H/E axes used in normalized reconstruction.
                        c = np.maximum(np.linalg.lstsq(REFERENCE_STAINS, od[mask].T,
                                                      rcond=None)[0], 0.0)
                        if not np.isfinite(c).all():
                            raise ValueError('Nonfinite calibration concentrations')
                        p95 = np.percentile(c, 95, axis=1)
                        if np.any(p95 <= EPSILON):
                            continue
                        records.append({'slide_file': str(source.relative_to(ROOT)),
                                        'slide_size_bytes': source.stat().st_size,
                                        'x': x, 'y': y, 'level': 0, 'band': band_number,
                                        'tissue_percent': float(percent),
                                        'tissue_pixels': int(mask.sum()),
                                        'concentration_p95': p95.tolist()})
                        accepted.append((x, y))
                        break
                    else:
                        raise ValueError('No valid reference patch in %s band %d' %
                                         (source.name, band_number))
                print(source.name, 'reference patches:', len(accepted))
        reference = np.median([r['concentration_p95'] for r in records], axis=0)
        config = {'schema_version': 1, 'stain_order': ['hematoxylin', 'eosin'],
                  'reference_percentile': reference.tolist(), 'percentile': 95,
                  'reference_stain_matrix': REFERENCE_STAINS.tolist(),
                  'seed': SEED, 'patch_size': 224, 'patch_count': len(records),
                  'selection': 'Eight row-ordered thumbnail tissue bands per slide; seeded sampling; '
                               'nonoverlapping level-0 patches; >=70% tissue; >=100 OD-mask pixels; '
                               'both concentration percentiles positive; max eight attempts per band.',
                  'calculation': 'Median of patchwise 95th-percentile nonnegative least-squares '
                                 'concentrations in the fixed reference reconstruction stain basis.',
                  'adaptive_modification': "C_prime = C * (P_ref / (P_source + epsilon)) ** alpha",
                  'alpha': 'clip(1 / (1 + mean(std(C, axis=1))), 0.5, 1.0); computed per source patch',
                  'scope': 'Pilot reference from available local slides; not population validated.',
                  'numpy_version': np.__version__, 'patches': records}
        REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
        REFERENCE_PATH.write_text(json.dumps(config, indent=2, allow_nan=False) + '\n')
        print('P_ref [H, E]:', reference.tolist())
        print('Saved:', REFERENCE_PATH)
        print('RuntimeWarnings: 0 (treated as errors)')


if __name__ == '__main__':
    main()
