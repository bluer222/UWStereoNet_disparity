import os

left_path = "leftImg8bit"
right_path = "rightImg8bit"

types = ["test", "train", "val"]
images = {}

destination = "data/cityscapes/{type}/{side}"

#get all image paths for each type
for t in types:
    images[t] = []
    for root, dirs, files in os.walk(os.path.join(left_path, t)):
        for file in files:
            full_path = os.path.join(root, file)
            #strip path to fetch relative path from leftImg8bit/train
            #eg leftImg8bit/train/stuttgart/stuttgart_000139_000019_leftImg8bit.png
            #to stuttgart/stuttgart_000139_000019_ (we drop the _leftImg8bit.png part)
            relative_path = os.path.relpath(full_path, os.path.join(left_path, t))
            #drop the _leftImg8bit.png part
            relative_path = relative_path.replace("_leftImg8bit.png", "")
            images[t].append(relative_path)

    print(f"{t} images: {len(images[t])}")

#move images into uwstereo input structure
i = 0
for image_subset in images:
    #create destination directories if they don't exist
    for side in ["left", "right"]:
        os.makedirs(destination.format(type=image_subset, side=side), exist_ok=True)

    for image in images[image_subset]:
        #generate source directory for left and right
        left_source = os.path.join(left_path, image_subset, image + "_leftImg8bit.png")
        right_source = os.path.join(right_path, image_subset, image + "_rightImg8bit.png")

        #generate destination directory for left and right
        left_destination = os.path.join(destination.format(type=image_subset, side="left"), str(i) + ".png")
        right_destination = os.path.join(destination.format(type=image_subset, side="right"), str(i) + ".png")
        i += 1

        #print(f"Copying {left_source} to {left_destination}")
        #print(f"Copying {right_source} to {right_destination}")
        #move images from source to destination
        os.system(f"mv {left_source} {left_destination}")
        os.system(f"mv {right_source} {right_destination}")
