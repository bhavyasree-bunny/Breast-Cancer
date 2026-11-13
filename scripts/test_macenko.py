"""Bounded regression test: synthetic cases and three patches per existing slide."""
from pathlib import Path
import warnings

import numpy as np
import openslide

from image_preprocessing import (modified_macenko, _estimate_stains, _contract,
                                 REFERENCE_STAINS, tissue_percentage)


def check_patch(patch):
    result = modified_macenko(patch)
    assert result.shape == patch.shape and result.dtype == np.uint8
    assert np.isfinite(result).all()
    assert np.array_equal(result, modified_macenko(patch))
    return result


def main():
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        for color in ([255, 255, 255], [0, 0, 0], [130, 60, 100]):
            patch = np.full((224, 224, 3), color, dtype=np.uint8)
            check_patch(patch)
        blank = np.full((224, 224, 3), 255, dtype=np.uint8)
        blank[0, :99] = [130, 60, 100]
        assert np.array_equal(check_patch(blank), blank)
        for bad in (np.nan, np.inf, -np.inf, -1, 256):
            patch = blank.astype(float)
            patch[0, 0, 0] = bad
            try:
                modified_macenko(patch)
            except ValueError:
                pass
            else:
                raise AssertionError('Invalid input was accepted')
        for tissue in (np.tile([0.5, 0.8, 0.4], (200, 1)),
                       np.linspace(0.1, 2, 200)[:, None] * [[0.5, 0.8, 0.4]]):
            assert np.array_equal(_estimate_stains(tissue), REFERENCE_STAINS)
        rng = np.random.default_rng(42)
        concentrations = rng.uniform(0.2, 2, (2, 224 * 224))
        od = _contract(REFERENCE_STAINS, concentrations)
        patch = (255 * np.exp(-od.T.reshape(224, 224, 3))).astype(np.uint8)
        patch[:10] = 255
        result = check_patch(patch)
        assert np.array_equal(result[:10], patch[:10])
        assert not np.array_equal(result[10:], patch[10:])
        estimated = _estimate_stains(od.T)
        assert not np.array_equal(estimated, REFERENCE_STAINS), 'Healthy estimate fell back'
        print('Synthetic validation passed (including degenerate stains and invalid RGB).')

        root = Path(__file__).resolve().parents[1]
        for name in ('slide1', 'slide2', 'slide3'):
            path = root / 'data' / 'images' / name / (name + '.svs')
            if not path.exists():
                print('SKIP missing:', path)
                continue
            count = 0
            with openslide.OpenSlide(str(path)) as slide:
                thumb = np.array(slide.get_thumbnail((512, 512)).convert('RGB'))
                ys, xs = np.where((thumb.mean(axis=2) < 210)
                                  & (np.ptp(thumb, axis=2) > 30))
                for index in np.linspace(0, len(xs) - 1, min(50, len(xs)), dtype=int):
                    x = min(int(xs[index] * slide.dimensions[0] / thumb.shape[1]),
                            slide.dimensions[0] - 224)
                    y = min(int(ys[index] * slide.dimensions[1] / thumb.shape[0]),
                            slide.dimensions[1] - 224)
                    patch = np.array(slide.read_region((x, y), 0, (224, 224)).convert('RGB'))
                    if tissue_percentage(patch) < 50:
                        continue
                    check_patch(patch)
                    count += 1
                    print(name, 'passed 224x224 tissue patch at', (x, y))
                    if count == 3:
                        break
            assert count == 3, 'Could not locate three tissue patches in ' + name
        print('PASS: no RuntimeWarnings (treated as errors); no large-scale processing.')


if __name__ == '__main__':
    main()
