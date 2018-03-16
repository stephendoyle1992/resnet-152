import csv
import numpy as np
import sys
import argparse
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("-f", "--csv_file", required=True, help="path to csv file")
    argp.add_argument("-x", "--img_width", required=True, help="width of original image")
    argp.add_argument("-y", "--img_height", required=True, help="height of original image")
    
    args = vars(argp.parse_args())

    img = Image.new('F', (int(args["img_width"]), int(args["img_height"])), 0)

    with open(args["csv_file"], 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            label = str(row[1])
            if label == 'Z':
                score = float(row[2])
                #divisor = float(row[3])
                posx = float(row[4])
                posy = float(row[5])
                #posx2 = float(row[6])
                new_width = float(row[7])
                #new_width2 = float(row[8])
                new_height = float(row[9])
                #new_height2 = float(row[10])
                #posy2 = float(row[11])
                ImageDraw.Draw(img).polygon(((posx, posy), (posx,new_width),
                    (new_width, new_height), (new_height, posy)), fill=score)

        myimg = np.ma.masked_equal(np.array(img), 0.)
        plt.imshow(myimg, interpolation="nearest")
        plt.colorbar()
        plt.show()

if __name__ == "__main__":
    main()