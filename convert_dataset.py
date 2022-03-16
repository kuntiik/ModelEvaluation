import os
import glob
import random
from shutil import copyfile


def check_if_directory_exist_or_create_it(path):
    if not os.path.isdir(path):
        print("INFO: Path does not exist.")
        try:
            print(path)
            os.mkdir(path)
            print("INFO: Directory created.")
        except OSError:
            print(f"ERROR: Directory {path} could not be created.")
            return 1
    return 0


def check_number_of_files_in_folder(path):
    return len(
        glob.glob(path + "/*")
    )  # len([name for name in os.listdir(path + "/.") if os.path.isfile(name)])


def change_suffix(file, suffix):
    base = os.path.splitext(file)[0]
    return base + suffix


def make_yolo_data(data_path, output_path):
    output_labels_path = os.path.join(output_path, "labels")
    output_images_path = os.path.join(output_path, "images")
    if not os.path.exists(output_labels_path):
        os.makedirs(output_labels_path)
    if not os.path.exists(output_images_path):
        os.makedirs(output_images_path)
    image_path = os.path.join(data_path, "obj_train_data")
    image_number = 0
    with open(os.path.join(output_path, "train.txt"), "w") as train, open(
        os.path.join(output_path, "dev.txt"), "w"
    ) as dev:
        for file in os.listdir(image_path):
            if file.endswith(".png"):
                if random.uniform(0, 1.0) <= 0.2:
                    dev.write("data/custom/images/" + str(image_number) + ".png\n")
                else:
                    train.write("data/custom/images/" + str(image_number) + ".png\n")
                copyfile(
                    os.path.join(image_path, file),
                    output_images_path + "/" + str(image_number) + ".png",
                )

                with open(
                    os.path.join(output_labels_path, str(image_number) + ".txt"), "w"
                ) as label, open(
                    os.path.join(image_path, change_suffix(file, ".txt")), "r"
                ) as label_unmodified:
                    lines = label_unmodified.readlines()
                    for line in lines:
                        if int(line[0]) < 1:
                            label.write(line)

                image_number += 1
    print("Modified data to yolo format")


if __name__ == "__main__":
    make_yolo_data("/home/kuntik/dev/rtg_data1", 0.2, "/home/kuntik/dev/rentgen_yolo")
