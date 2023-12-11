import os
from PIL import Image

# map folder -> path to vol image
datas = {
    "images":"data/train-volume.tif",
    "masks":"data/train-labels.tif",
    "test":"data/test-volume.tif"
}

for folder, path in datas.items():
    # Open the volume image
    folder_path = f"data/{folder}"

    # Delete the folder if it exists
    # shutil.rmtree(folder_path, ignore_errors=True)
    
    # Create the folder
    os.makedirs(folder_path, exist_ok=True)

    with Image.open(path) as volume_img:
        # Loop through each page in the image
        for i in range(volume_img.n_frames):
            # Select the current page
            volume_img.seek(i)
            # Convert the page to a separate image
            page_img = volume_img.copy()
            # Save the image with a unique filename
            page_img.save(f"data/{folder}/{folder.removesuffix('s')}_{i}.jpg", "JPEG")
            

print("Data setup complete.")