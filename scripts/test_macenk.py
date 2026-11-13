import os
import numpy as np
import openslide
import matplotlib.pyplot as plt

from image_preprocessing import modified_macenko


SVS_PATH = "data/images/slide1/slide1.svs"

OUTPUT_DIR = "data/macenko_test"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# Open SVS
slide = openslide.OpenSlide(SVS_PATH)

print("Slide dimensions:", slide.dimensions)


# Read a small region from the slide
patch = slide.read_region(
    (10000, 10000),
    0,
    (224, 224)
).convert("RGB")

slide.close()


# Convert to NumPy
original = np.array(patch)


# Apply modified Macenko
normalized = modified_macenko(
    original
)


# Save original
plt.imsave(
    os.path.join(
        OUTPUT_DIR,
        "original.png"
    ),
    original
)


# Save normalized
plt.imsave(
    os.path.join(
        OUTPUT_DIR,
        "modified_macenko.png"
    ),
    normalized
)


print("Original saved:")
print(
    "data/macenko_test/original.png"
)

print("Normalized saved:")
print(
    "data/macenko_test/modified_macenko.png"
)