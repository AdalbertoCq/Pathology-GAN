import matplotlib.pyplot as plt
import glob
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


def digits(image_path):
    dig = image_path.split('/')[-1]
    dig = dig.split('_')[0]
    return dig


def draw_text_img(cropped_img, text, run_i, base_width, image_i):
    draw = ImageDraw.Draw(cropped_img)
    font = ImageFont.truetype('/opt/X11/share/fonts/TTF/luximb.ttf', 18)
    if run_i == 0:
        draw.text((0, 0), 'Original',(0,0,0), font=font)
        draw.text((0, 25), 'Iter %s' % image_i,(0,0,0), font=font)
        draw.text((base_width, 0), text,(0,0,0), font=font)
    else:
        draw.text((0, 0), text,(0,0,0), font=font)
        
    
def plot_samples(main_dic, min_set, path_runs):
    one_run = main_dic['runs'][0]
    one_image = main_dic['images'][0]
    one_image_d = Image.open('%s/%s' % (one_run, one_image))
    width, height = one_image_d.size
    base_width = width/2
    
    n_rows = len(main_dic['images'])
    total_height = min_set * height
    total_width = (len(main_dic['runs'])-1) * base_width + width
    
    new_im = Image.new('RGB', (int(total_width), total_height))
    index_row = 0
    for image_i, image in enumerate(main_dic['images']):
        index_col = 0
        for run_i, run_path in enumerate(main_dic['runs']):
            path_img = '%s/%s' % (run_path, image)
            image_raw = Image.open(path_img)
            cropped_img = image_raw
            if run_i != 0:
                area = (base_width, 0, width, height)
                cropped_img = image_raw.crop(area)
            draw_text_img(cropped_img, run_path.split('/')[-1], run_i, base_width, image_i)
            new_im.paste(cropped_img, (index_col, index_row))
            index_col += cropped_img.size[0]
        index_row += cropped_img.size[1]
    plt.show(new_im)
    new_im.save('%s/run/comparison.png' % path_runs)


def get_images(path_runs):
    runs = glob.glob('%s/run/*' % path_runs)
    main_dic = dict()
    min_set = 100000
    main_dic['runs'] = list()
    for run_main_path in runs:
        images = glob.glob('%s/img/*.png' % run_main_path)
        if len(images) == 0:
            continue
        main_dic['runs'].append(run_main_path)
        f_images = list()
        for image in images:
            if 'gen_' in image:
                continue
            image = image.split(run_main_path)[-1]
            if image.startswith('/'):
                image = image[1:]
            f_images.append(image)
        if len(f_images) < min_set:
            min_set = len(f_images)
            main_dic['images'] = sorted(f_images, key=lambda f: int(digits(f)))

    return main_dic, min_set


path_runs = '/Users/adalbertoclaudioquiros/Documents/Code/UofG/PhD/Cancer TMA Generative'

main_dict, min_set = get_images(path_runs)
plot_samples(main_dict, min_set, path_runs)








