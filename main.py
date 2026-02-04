import os
import glob
import argparse
from PIL import Image
import torch
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.pipelines.rembg import BiRefNet
import o_voxel

# 1. Configuration
INPUT_DIR = "inputs"
PROCESSED_DIR = "processed"
OUTPUT_DIR = "output"
ALLOWED_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.webp")

# Mesh Export Configuration
CONFIG_TEXTURE_SIZE = 2048
CONFIG_DECIMATION_TARGET = 1000000

# Generation Configuration
# Resolution Options: '512', '1024', '1024_cascade', '1536_cascade'
CONFIG_RESOLUTION = '1024_cascade'
CONFIG_PREPROCESS = False  # Background removal is handled separately
CONFIG_LOW_VRAM = True     # Enables sequential model offloading to CPU

# Ensure environment variables for GPU performance
import gc
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def set_resolution(resolution_str):
    """Parse and set the resolution configuration"""
    global CONFIG_RESOLUTION
    if resolution_str:
        resolution_input = resolution_str.strip()
        try:
            res_num = int(resolution_input)
            if res_num == 512:
                CONFIG_RESOLUTION = '512'
                return True, "512 (~8-10GB VRAM)"
            elif res_num == 1024:
                CONFIG_RESOLUTION = '1024'
                return True, "1024 (~12-14GB VRAM)"
            else:
                return False, f"Invalid resolution '{res_num}'. Valid options: 512, 1024"
        except ValueError:
            return False, f"Could not parse resolution '{resolution_input}'. Valid options: 512, 1024"
    return False, None

def remove_backgrounds():
    """Remove backgrounds from images in INPUT_DIR and save to PROCESSED_DIR"""
    # Setup directories
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Find input images
    image_paths = []
    for ext in ALLOWED_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(INPUT_DIR, ext)))

    if not image_paths:
        print(f"No images found in {INPUT_DIR}")
        return

    # Check which images need processing
    images_to_process = []
    for img_path in image_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(PROCESSED_DIR, f"{base_name}.png")
        if not os.path.exists(output_path):
            images_to_process.append(img_path)

    if not images_to_process:
        print(f"\n=== Background Removal Stage ===")
        print(f"All {len(image_paths)} images already processed. Skipping.")
        return

    print(f"\n=== Background Removal Stage ===")
    print(f"Found {len(image_paths)} images, {len(images_to_process)} need processing.")

    # Load background removal model
    print("Loading BiRefNet for background removal...")
    birefnet = BiRefNet()
    birefnet.to("cuda")

    # Process each image
    for idx, img_path in enumerate(images_to_process, 1):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(PROCESSED_DIR, f"{base_name}.png")

        print(f"[{idx}/{len(images_to_process)}] Removing background: {img_path}")
        try:
            image = Image.open(img_path).convert("RGB")
            image_no_bg = birefnet(image)
            image_no_bg.save(output_path)
            print(f"  -> Saved to: {output_path}")
        except Exception as e:
            print(f"  X Failed: {e}")

    # Cleanup model after all images processed
    del birefnet
    torch.cuda.empty_cache()
    gc.collect()

    print(f"\n=== Background Removal Complete ===\n")

def process_images():
    """Generate 3D models from processed images"""
    # 2. Setup Directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 3. Find Processed Images
    image_paths = []
    for ext in ALLOWED_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(PROCESSED_DIR, ext)))

    if not image_paths:
        print(f"\n=== 3D Model Generation Stage ===")
        print(f"No images found in {PROCESSED_DIR}")
        return

    # Check which images need processing
    images_to_process = []
    for img_path in image_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(OUTPUT_DIR, f"{base_name}.glb")
        if not os.path.exists(output_path):
            images_to_process.append(img_path)

    if not images_to_process:
        print(f"\n=== 3D Model Generation Stage ===")
        print(f"All {len(image_paths)} models already generated. Skipping.")
        return

    print(f"\n=== 3D Model Generation Stage ===")
    print(f"Found {len(image_paths)} images, {len(images_to_process)} need processing.")

    # Load Pipeline
    print("Loading Trellis2 pipeline...")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.low_vram = CONFIG_LOW_VRAM
    pipeline.to("cuda")

    # Ensure samplers don't store history to save VRAM
    pipeline.sparse_structure_sampler_params['save_history'] = False
    pipeline.shape_slat_sampler_params['save_history'] = False
    pipeline.tex_slat_sampler_params['save_history'] = False

    # 5. Process Each Image
    for idx, img_path in enumerate(images_to_process, 1):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(OUTPUT_DIR, f"{base_name}.glb")

        print(f"\n[{idx}/{len(images_to_process)}] Processing: {img_path}")
        try:
            image = Image.open(img_path)

            # Run the workflow
            print(f"  -> Generating 3D structure...")
            results = pipeline.run(
                image,
                pipeline_type=CONFIG_RESOLUTION,
                preprocess_image=CONFIG_PREPROCESS
            )
            mesh = results[0]

            print(f"  -> Structure complete")

            # Simplify if needed
            mesh.simplify(16777216) # nvdiffrast limit

            # Export to GLB
            print(f"  -> Exporting mesh to GLB...")
            glb = o_voxel.postprocess.to_glb(
                vertices            =   mesh.vertices,
                faces               =   mesh.faces,
                attr_volume         =   mesh.attrs,
                coords              =   mesh.coords,
                attr_layout         =   mesh.layout,
                voxel_size          =   mesh.voxel_size,
                aabb                =   [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target   =   CONFIG_DECIMATION_TARGET,
                texture_size        =   CONFIG_TEXTURE_SIZE,
                remesh              =   True,
                remesh_band         =   1,
                remesh_project      =   0,
                verbose             =   False
            )
            glb.export(output_path, extension_webp=True)

            print(f"  OK Saved: {output_path}")

        except Exception as e:
            print(f"  X Failed: {e}")
        finally:
            # Cleanup after each image (3D generation is memory-intensive)
            torch.cuda.empty_cache()
            gc.collect()

    # Final cleanup
    del pipeline
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TRELLIS.2 3D Model Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run full pipeline (background removal + 3D generation)
  python main.py --background       # Only remove backgrounds from input images
  python main.py --generate         # Only generate 3D models from processed images
  python main.py --resolution 512   # Use 512 resolution (lower VRAM)
  python main.py --resolution 1024  # Use 1024 resolution (medium quality)
  python main.py -r 1024            # Short form
        """
    )
    parser.add_argument(
        "--background", "-b",
        action="store_true",
        help="Only run background removal step (inputs/ -> processed/)"
    )
    parser.add_argument(
        "--generate", "-g",
        action="store_true",
        help="Only run 3D model generation step (processed/ -> output/)"
    )
    parser.add_argument(
        "--resolution", "-r",
        type=str,
        default=None,
        help="Set resolution (512 or 1024). Default: 1024_cascade from config"
    )

    args = parser.parse_args()

    # Parse and set resolution
    if args.resolution:
        success, message = set_resolution(args.resolution)
        if success:
            print(f"Using resolution: {message}")
        else:
            print(f"Warning: {message}")
            print(f"Falling back to default: {CONFIG_RESOLUTION}")

    # Determine which steps to run
    run_background = args.background or (not args.background and not args.generate)
    run_generate = args.generate or (not args.background and not args.generate)

    # Execute requested steps
    if run_background:
        remove_backgrounds()

    if run_generate:
        process_images()
