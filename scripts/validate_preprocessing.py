"""Small end-to-end SVS validation. Run with Python 3.9 or later."""
import csv
import json
from pathlib import Path
import warnings

import numpy as np
import openslide
from PIL import Image

from image_preprocessing import modified_macenko, tissue_percentage

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'data' / 'preprocessing_validation'
PATCH_SIZE = 224
PATCHES_PER_SLIDE = 3
MAX_CANDIDATES = 30


def validate_saved(path):
    if not path.is_file():
        raise ValueError('Missing patch: ' + str(path))
    with Image.open(path) as image:
        image.load()
        if image.size != (PATCH_SIZE, PATCH_SIZE) or image.mode != 'RGB':
            raise ValueError('Invalid patch dimensions or mode: ' + str(path))
        if not np.isfinite(np.asarray(image)).all():
            raise ValueError('Nonfinite saved pixels: ' + str(path))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    report = {'slides_processed': 0, 'patches_per_slide': {},
              'failed_patches': 0, 'rejected_background_patches': 0,
              'errors': [], 'missing_slides': [], 'output_directory': str(OUTPUT)}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for name in ('slide1', 'slide2', 'slide3'):
            source = ROOT / 'data' / 'images' / name / (name + '.svs')
            if not source.is_file():
                report['missing_slides'].append(str(source.relative_to(ROOT)))
                continue
            count = 0
            try:
                with openslide.OpenSlide(str(source)) as slide:
                    report['slides_processed'] += 1
                    # A small thumbnail locates candidates; every saved patch is
                    # read directly at level 0, never from a full-slide PNG.
                    thumb = np.asarray(slide.get_thumbnail((512, 512)).convert('RGB'))
                    ys, xs = np.where((thumb.mean(axis=2) < 210)
                                      & (np.ptp(thumb, axis=2) > 30))
                    seen = set()
                    for index in np.linspace(0, len(xs) - 1,
                                             min(MAX_CANDIDATES, len(xs)), dtype=int):
                        x = min(int(xs[index] * slide.dimensions[0] / thumb.shape[1]),
                                slide.dimensions[0] - PATCH_SIZE)
                        y = min(int(ys[index] * slide.dimensions[1] / thumb.shape[0]),
                                slide.dimensions[1] - PATCH_SIZE)
                        if (x, y) in seen:
                            continue
                        seen.add((x, y))
                        try:
                            patch = np.asarray(slide.read_region(
                                (x, y), 0, (PATCH_SIZE, PATCH_SIZE)).convert('RGB'))
                            percent = tissue_percentage(patch)
                            if percent < 50:
                                report['rejected_background_patches'] += 1
                                continue
                            normalized = modified_macenko(patch)
                            if (normalized.shape != (PATCH_SIZE, PATCH_SIZE, 3)
                                    or normalized.dtype != np.uint8
                                    or not np.isfinite(normalized).all()):
                                raise ValueError('Invalid normalization output')
                            destination = OUTPUT / (name + '_x%d_y%d.png' % (x, y))
                            Image.fromarray(normalized).save(destination)
                            validate_saved(destination)
                            rows.append({'slide_file': str(source.relative_to(ROOT)),
                                         'patch_file': str(destination.relative_to(ROOT)),
                                         'tissue_percent': '%.6f' % percent,
                                         'normalization_method': 'modified_macenko'})
                            count += 1
                        except Exception as error:
                            report['failed_patches'] += 1
                            report['errors'].append('%s (%d,%d): %s' % (name, x, y, error))
                        if count == PATCHES_PER_SLIDE:
                            break
            except Exception as error:
                report['errors'].append(name + ': ' + str(error))
            report['patches_per_slide'][name] = count
            if count != PATCHES_PER_SLIDE:
                report['errors'].append(name + ': fewer than three verified patches')
        manifest = OUTPUT / 'patches.csv'
        with manifest.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                'slide_file', 'patch_file', 'tissue_percent', 'normalization_method'])
            writer.writeheader()
            writer.writerows(rows)
        # Recheck the persisted manifest, not just the in-memory outputs.
        with manifest.open(newline='') as handle:
            saved_rows = list(csv.DictReader(handle))
        for row in saved_rows:
            validate_saved(ROOT / row['patch_file'])
        report['verified_patches'] = len(saved_rows)
    report['runtime_warnings'] = sum(issubclass(w.category, RuntimeWarning) for w in caught)
    report['warnings'] = [str(w.message) for w in caught]
    (OUTPUT / 'validation_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if report['errors'] or report['runtime_warnings'] or not rows:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
