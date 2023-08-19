import copy
from pprint import pprint
from imageio import imread
import numpy
import glob
from numpy.core.multiarray import ndarray

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--max_chars",
                    help="maximum number of characters to use",
                    type=int)
parser.add_argument("--multiplicity",
                    help="how many times to repeat each character in the dataset",
                    type=int, default=6)
args = parser.parse_args()


def gen_letter_image(letter, shiftx, shifty, fontname, size):
    from PIL import Image, ImageDraw, ImageFont
    image = Image.new('RGB', (64, 64), color='black')
    try:
        fontfile = 'fonts/%s.ttf' % fontname
        font = ImageFont.truetype(fontfile, size=size)
    except:
        import wget

        if fontname == 'Chinese':
            print('Beginning font download')
            url = 'https://drive.google.com/uc?export=download&id=1QBW8KLl0JG92C7V72sf7vlDz73PxbHFx'
            wget.download(url, fontfile)
        else:
            print('Font not available.')
            raise
        font = ImageFont.truetype(fontfile, size=size)

    draw = ImageDraw.Draw(image)
    draw.text((shiftx, shifty), '%s' % (letter), fill=(255, 255, 255), font=font)
    folder = './temp/'

    max_value = numpy.amax(numpy.array(image))

    if max_value>0:
        import os
        if not os.path.exists(folder):
            os.mkdir(folder)
        filename = '%s/%s.png' % (folder, letter)
        image.save(filename)


def gen_letter_image_chinese(letter):
    gen_letter_image(letter, 7, -1, "Chinese", 50)


def apply_noise(image, sigma):
    noise = numpy.random.normal(0, 1, image.shape)
    #coords = [numpy.random.randint(0, j - 1, int(image.size // 2)) for j in image.shape]
    #image_holes = numpy.copy(image)
    #image_holes[tuple(coords)] = 0
    #return numpy.clip(image_holes + sigma * noise, 0, 1)
    return image + sigma * noise


def apply_pep_pg52_noise(image, sigma):
    noise = numpy.random.normal(0, 1, image.shape) * 0.04
    image_ = 0.05*numpy.random.poisson(image, image.shape) + sigma * noise
    coords = [numpy.random.randint(0, j - 1, int(image.size // 2)) for j in image.shape]
    image_holes = numpy.copy(image_)
    image_holes[tuple(coords)] = 0
    return image_holes 


def apply_pep_noise(image, sigma):
    coords = [numpy.random.randint(0, j - 1, int(image.size // 2)) for j in image.shape]
    image_holes = numpy.copy(image)
    image_holes[tuple(coords)] = 0
    return image_holes


def apply_deg_noise(image, dtype):
    row, line = image.shape
    image_ = numpy.copy(image)
    if dtype == 'mean':  # 0.5514455  -3.0238558e-09
        image_part = numpy.reshape(image_[: row-row % 3, : line-line % 3],
                               (row//3, 3, line//3, 3))
        image_part[...] = image_part.mean((1, 3), keepdims=True)
    if dtype == 'median':  # 0.55670434  -0.003725313
        image_part = numpy.reshape(image_[: row - row % 3, : line - line % 3],
                                   (row // 3, 3, line // 3, 3))
        image_part[...] = numpy.median(image_part, (1, 3), keepdims=True)
    if dtype == 'center':  # 0.56566054  -0.00235
        image_part = numpy.reshape(image_[: row - row % 3, : line - line % 3],
                                   (row // 3, 3, line // 3, 3))
        image_part[...] = image_part[:, 1:2, :, 1:2]
    image_[: row - row % 3, : line - line % 3] = numpy.reshape(image_part, (row - row % 3, line - line % 3))

    return image_


def apply_lap_noise(image, sigma):
    noise = numpy.random.laplace(0, 1, image.shape) / 2**0.5
    return image + sigma * noise

def apply_poi_noise(image, sigma):
    return numpy.random.poisson(image, image.shape) 

def apply_pg_noise(image, sigma):
    noise = numpy.random.normal(0, 1, image.shape)
    return numpy.random.poisson(image, image.shape) + sigma * noise    

def apply_pg52_noise(image, sigma):
    noise = numpy.random.normal(0, 1, image.shape) * 0.04
    return 0.05*numpy.random.poisson(image, image.shape) + sigma * noise

def apply_log_noise(image, sigma):
    noise = numpy.random.logistic(0, 3**0.5/numpy.pi, image.shape)
    return image + sigma*noise

def apply_uni_noise(image, sigma):
    noise = numpy.random.uniform(-3**0.5, 3**0.5, image.shape)
    return image + sigma * noise

def apply_mix_noise(image, sigma):
    noise = (numpy.random.logistic(0, 3**0.5/numpy.pi, image.shape) +
            #numpy.random.laplace(0, 1, image.shape)/2**0.5 +
            numpy.random.uniform(-1, 1, image.shape)*3**0.5 +
            numpy.random.normal(0, 1, image.shape))
    return image + sigma/3**0.5 * noise

sigma1 = 0.3
sigma2 = 0.5
sigma3 = 0.7
sigma4 = 0.9

def generate_tiles(rounds=10, sigma1=sigma1, sigma2=sigma2, sigma3=sigma3):
    list = []
    for i in range(rounds):
        for image_path in glob.glob("./temp/*.png"):
            image_gt = imread(image_path)[:, :, 0].astype(numpy.float32) / 255
            image_ns1: ndarray = apply_deg_noise(image_gt, 'mean')
            image_ns2: ndarray = apply_deg_noise(image_gt, 'median')
            image_ns3: ndarray = apply_deg_noise(image_gt, 'center')
            image_ns4: ndarray = apply_deg_noise(image_gt, 'center')

            image_ns1b: ndarray = apply_deg_noise(image_gt, 'mean')
            image_ns2b: ndarray = apply_deg_noise(image_gt, 'median')
            image_ns3b: ndarray = apply_deg_noise(image_gt, 'center')
            image_ns4b: ndarray = apply_deg_noise(image_gt, 'center')

            tile = numpy.stack([image_gt[:, :],
                                     image_ns1[:, :],
                                     image_ns2[:, :],
                                     image_ns3[:, :],
                                     image_ns4[:, :],
                                     image_ns1b[:, :],
                                     image_ns2b[:, :],
                                     image_ns3b[:, :],
                                     image_ns4b[:, :]
                                ])
            list.append(tile[numpy.newaxis])

    return list

def clear_temp():
    import os
    import glob
    files = glob.glob('./temp/*.png')
    for f in files:
        os.remove(f)

def training(max_chars=None, multiplicity=6):
    print("Clearing Temp... ")
    clear_temp()

    print("Generating png... ")
    from hanzichars import ChineseChars
    for i, c in enumerate(ChineseChars):
        gen_letter_image_chinese(c)
        if max_chars and i > max_chars:
            break

    print("Generating tiles... ")
    tilelist = generate_tiles(multiplicity)
    print("Total number of tiles: %d " % len(tilelist))

    print("Stacking...")
    tilesarray = numpy.vstack(tilelist)

    print("Shuffling...")
    numpy.random.shuffle(tilesarray)

    print("Shape:")
    print(tilesarray.shape)

    print("Splitting...")

    nbtiles = tilesarray.shape[0]
    trainendidx = int(nbtiles * 0.7)
    valendidx = int(nbtiles * 0.9)

    training_dataset = tilesarray[:trainendidx]
    validation_dataset = tilesarray[trainendidx:valendidx]
    test_dataset = tilesarray[valendidx:]

    print("Saving tiles...")
    numpy.save("./tiles/train_deg", training_dataset)
    numpy.save("./tiles/val_deg", validation_dataset)
    numpy.save("./tiles/test_deg", test_dataset)

def test():
    print("Loading tiles...")
    testing = numpy.load("./tiles/test.npy")
    pprint(testing.shape)

    print("Generating images...")
    images_list = []
    nb_images = 100
    nb_images = min(len(testing)//64, nb_images)
    padding = False

    counter = 0
    for image_index in range(0,nb_images):
        for image_type in range(0,5):
            image_list_y = []
            for iy in range(0,8):
                image_list_x = []
                for ix in range(0, 8):
                    image_gt = copy.deepcopy(testing[counter+ix+8*iy, 0])

                    if padding:
                        pad_size = image_gt.shape[0] // 4
                        image_gt = numpy.pad(image_gt,
                                          ((pad_size, pad_size), (pad_size, pad_size)),
                                          'constant',
                                          constant_values=((0, 0), (0, 0)))

                    if image_type==0:
                        image = image_gt
                    elif image_type==1:
                        image = apply_pep_pg52_noise(image_gt, sigma1)
                    elif image_type == 2:
                        image = apply_pep_pg52_noise(image_gt, sigma2)
                    elif image_type == 3:
                        image = apply_pep_pg52_noise(image_gt, sigma3)
                    elif image_type == 4:
                        image = apply_pep_pg52_noise(image_gt, sigma4)

                    image_list_x.append(image)
                array_x = numpy.concatenate(tuple(image_list_x), axis=0)
                image_list_y.append(array_x)
            testimage = numpy.concatenate(tuple(image_list_y), axis=1)

            images_list.append(testimage[numpy.newaxis])

        counter = counter + 64

    images = numpy.concatenate(tuple(images_list), axis=0)
    pprint(images.shape)

    images = numpy.reshape(images,(nb_images,5, images.shape[1], images.shape[2]))
    pprint(images.shape)

    numpy.save("./fulltest/testimages" , images)


if __name__ == '__main__':
    training(max_chars=args.max_chars, multiplicity=args.multiplicity)
    # test()

exit(0)
