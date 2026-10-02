"""
Generates synthetic sample test images for Forest Intelligence CV unit tests.
NOTE: These images are synthetic test data explicitly designed for automated testing
and must NOT be presented as real-world forest observations.
"""

from pathlib import Path
import numpy as np
from PIL import Image

def generate_sample_test_images(output_dir: Path) -> dict:
    """
    Creates synthetic test images labeled as sample test data:
    - Normal forest canopy (green/brown tones)
    - Forest fire sample (flame cluster on forest background)
    - Forest smoke sample (desaturated gray haze plume)
    - Combined fire & smoke sample
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(42)
    height, width = 200, 200

    created_paths = {}

    # 1. Normal Forest Image
    # Forest canopy greens: R in [20, 60], G in [100, 170], B in [20, 60]
    forest_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    forest_rgb[..., 0] = rng.randint(20, 60, (height, width), dtype=np.uint8)
    forest_rgb[..., 1] = rng.randint(100, 170, (height, width), dtype=np.uint8)
    forest_rgb[..., 2] = rng.randint(20, 60, (height, width), dtype=np.uint8)
    # Add some earth tones in bottom 25% (soil/trunks)
    forest_rgb[150:, :, 0] = rng.randint(80, 115, (50, width), dtype=np.uint8)
    forest_rgb[150:, :, 1] = rng.randint(55, 80, (50, width), dtype=np.uint8)
    forest_rgb[150:, :, 2] = rng.randint(25, 45, (50, width), dtype=np.uint8)
    
    normal_path = output_dir / "sample_test_normal_forest.png"
    Image.fromarray(forest_rgb).save(normal_path)
    created_paths["normal"] = normal_path

    # 2. Fire Sample Image
    # Base forest + bright orange-red fire patch in center
    fire_rgb = forest_rgb.copy()
    # 40x40 patch of fire: R in [220, 255], G in [80, 140], B in [5, 35]
    fire_patch_h, fire_patch_w = 50, 50
    start_y, start_x = 75, 75
    fire_rgb[start_y:start_y+fire_patch_h, start_x:start_x+fire_patch_w, 0] = rng.randint(220, 255, (fire_patch_h, fire_patch_w), dtype=np.uint8)
    fire_rgb[start_y:start_y+fire_patch_h, start_x:start_x+fire_patch_w, 1] = rng.randint(90, 145, (fire_patch_h, fire_patch_w), dtype=np.uint8)
    fire_rgb[start_y:start_y+fire_patch_h, start_x:start_x+fire_patch_w, 2] = rng.randint(10, 35, (fire_patch_h, fire_patch_w), dtype=np.uint8)
    
    fire_path = output_dir / "sample_test_fire.png"
    Image.fromarray(fire_rgb).save(fire_path)
    created_paths["fire"] = fire_path

    # 3. Smoke Sample Image
    # Base forest + grayish/whitish plume across the top/middle
    smoke_rgb = forest_rgb.copy()
    # Smoke plume: R, G, B close to each other in [170, 210]
    smoke_h, smoke_w = 60, 120
    sm_y, sm_x = 30, 40
    base_gray = rng.randint(175, 215, (smoke_h, smoke_w), dtype=np.uint8)
    smoke_rgb[sm_y:sm_y+smoke_h, sm_x:sm_x+smoke_w, 0] = base_gray + rng.randint(-3, 4, (smoke_h, smoke_w), dtype=np.int16).clip(0, 255).astype(np.uint8)
    smoke_rgb[sm_y:sm_y+smoke_h, sm_x:sm_x+smoke_w, 1] = base_gray + rng.randint(-3, 4, (smoke_h, smoke_w), dtype=np.int16).clip(0, 255).astype(np.uint8)
    smoke_rgb[sm_y:sm_y+smoke_h, sm_x:sm_x+smoke_w, 2] = base_gray + rng.randint(-3, 4, (smoke_h, smoke_w), dtype=np.int16).clip(0, 255).astype(np.uint8)

    smoke_path = output_dir / "sample_test_smoke.png"
    Image.fromarray(smoke_rgb).save(smoke_path)
    created_paths["smoke"] = smoke_path

    # 4. Combined Fire and Smoke Sample Image
    combined_rgb = smoke_rgb.copy()
    combined_rgb[start_y:start_y+fire_patch_h, start_x:start_x+fire_patch_w, 0] = rng.randint(220, 255, (fire_patch_h, fire_patch_w), dtype=np.uint8)
    combined_rgb[start_y:start_y+fire_patch_h, start_x:start_x+fire_patch_w, 1] = rng.randint(90, 145, (fire_patch_h, fire_patch_w), dtype=np.uint8)
    combined_rgb[start_y:start_y+fire_patch_h, start_x:start_x+fire_patch_w, 2] = rng.randint(10, 35, (fire_patch_h, fire_patch_w), dtype=np.uint8)
    
    combined_path = output_dir / "sample_test_fire_smoke.png"
    Image.fromarray(combined_rgb).save(combined_path)
    created_paths["fire_smoke"] = combined_path

    # 5. Empty (0-byte) file for negative validation test
    empty_path = output_dir / "sample_test_empty.png"
    empty_path.write_bytes(b"")
    created_paths["empty"] = empty_path

    # 6. Corrupt file for corruption testing
    corrupt_path = output_dir / "sample_test_corrupt.png"
    corrupt_path.write_bytes(b"NOT_A_REAL_IMAGE_DATA_CORRUPT_HEADER")
    created_paths["corrupt"] = corrupt_path

    return created_paths

if __name__ == "__main__":
    target = Path(__file__).resolve().parent.parent.parent / "data" / "sample"
    paths = generate_sample_test_images(target)
    print(f"Generated sample images in {target}:")
    for k, p in paths.items():
        print(f" - {k}: {p.name}")
