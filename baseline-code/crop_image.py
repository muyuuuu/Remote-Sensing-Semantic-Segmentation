import os
from PIL import Image
import numpy as np


def crop(img1, img2, img3):

    cnt = 1
    x = [0, 600, 1200]
    y = [0, 600, 1200]

    fold1, file_name = img1
    fold2, dem_name = img2
    fold3, label_file = img3

    for i in x:
        for j in y:

            label_image = Image.open(fold3 + label_file)
            s_i = i + 800
            s_j = j + 800

            if label_image.width < s_i:
                s_i = label_image.width
            if label_image.height < s_j:
                s_j = label_image.height

            cropped_label = label_image.crop((i, j, s_i, s_j))

            check = np.array(cropped_label)
            if (check == 0).sum() >= check.shape[0] * check.shape[1] * 7 / 8:
                continue

            origin_image = Image.open(fold1 + file_name)
            cropped_origin = origin_image.crop((i, j, s_i, s_j))

            dem_image = Image.open(fold2 + dem_name)
            cropped_dem = dem_image.crop((i // 2, j // 2, s_i // 2, s_j // 2))

            if 3 in check:
                cropped_origin.save("./view/3/" + str(cnt) + "_" + file_name)
                cropped_dem.save("./view/3/" + str(cnt) + "_" + dem_name)

            if 6 in check:
                cropped_origin.save("./view/6/" + str(cnt) + "_" + file_name)
                cropped_dem.save("./view/6/" + str(cnt) + "_" + dem_name)

            if 12 in check:
                cropped_origin.save("./view/12/" + str(cnt) + "_" + file_name)
                cropped_dem.save("./view/12/" + str(cnt) + "_" + dem_name)

            if 13 in check:
                cropped_origin.save("./view/13/" + str(cnt) + "_" + file_name)
                cropped_dem.save("./view/13/" + str(cnt) + "_" + dem_name)

            cnt += 1


host = r"C:\Users\muyu0\Downloads\labeled_train\\"

root = [host + "Nantes_Saint-Nazaire/", host + "Nice/"]
sub_folder = ["BDORTHO/", "RGEALTI/", "UrbanAtlas/"]

for r in root:
    folder1 = r + sub_folder[0]
    folder2 = r + sub_folder[1]
    folder3 = r + sub_folder[2]

    for file in os.listdir(folder1):
        file_name = file.split('.')[0]
        dem_name = file_name + "_RGEALTI" + ".tif"
        label_name = file_name + "_UA2012" + ".tif"
        file_name += ".tif"

        img1 = folder1 + file_name
        img2 = folder2 + dem_name
        img3 = folder3 + label_name

        crop([folder1, file_name], [folder2, dem_name], [folder3, label_name])
        # break
    # break
