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
    argp.add_argument('-b', "--bold", default="1",required=False, help="bold output (score above .500 fills X amount, else fill 1)")
    
    args = vars(argp.parse_args())

    img = Image.new('F', (int(args["img_width"]), int(args["img_height"])), 0)

    with open(args["csv_file"], 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            label = str(row[1])
            if label == 'Z':
                score = float(row[2])
                divisor = float(row[3])
                posx = float(row[4])
                posy = float(row[5])
                new_width = float(row[7])
                new_height = float(row[9])
                if args['bold'] == "1":
                    ImageDraw.Draw(img).polygon(((posx, posy), (posx,new_height),
                        (new_width, new_height), (new_width, posy)), fill=score)
                else:
                    if score > .500:
                        ImageDraw.Draw(img).polygon(((posx, posy), (posx,new_height),
                        (new_width, new_height), (new_width, posy)), fill=int(args['bold']))
                    else:
                        ImageDraw.Draw(img).polygon(((posx, posy), (posx,new_height),
                        (new_width, new_height), (new_width, posy)), fill=1)


    myimg = np.ma.masked_equal(np.array(img), 0.)
    plt.imshow(myimg, interpolation="nearest")
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    main()