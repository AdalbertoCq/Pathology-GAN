import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import numpy as np
import os
import shutil
import h5py
import tensorflow as tf
import csv
import json
import random
import math


# Simple function to plot number images.
def plot_images(plt_num, images, dim1=None, dim2=None, wspace=None, title=None, axis='off', plt_save=None):
    # Standard parameters for the plot.
    
    if dim1 is not None and dim2 is not None:
        fig = plt.figure(figsize=(dim1, dim2))
    else:
        fig = plt.figure()

    if wspace is not None:
        plt.subplots_adjust(wspace=wspace)
        
    if title is not None:
        fig.suptitle(title)

    for i in range(0, plt_num):
        fig.add_subplot(1, 10, i+1)
        img = images[i, :, :, :]
        plt.imshow(img)
        plt.axis(axis)
    if plt_save is not None:
        plt.savefig(plt_save)
    plt.show()


# Plot and save figure of losses.
def save_loss(losses, data_out_path, dim):    
    mpl.rcParams["figure.figsize"] = dim, dim
    plt.rcParams.update({'font.size': 22})
    losses = np.array(losses)
    num_loss = losses.shape[1]
    for _ in range(num_loss):
        if _ == 0:
            label = 'Generator'
        elif _ == 1:
            label = 'Discriminator'
        else:
            label = 'Mutual Information'
        plt.plot(losses[:, _], label=label, alpha=0.5)
    plt.title("Training Losses")
    plt.legend()
    plt.savefig('%s/training_loss.png' % data_out_path)


def get_checkpoint(data_out_path, which=0):
    checkpoints_path = os.path.join(data_out_path, 'checkpoints')
    checkpoints = os.path.join(checkpoints_path, 'checkpoint')
    index = 0
    with open(checkpoints, 'r') as f:
        for line in reversed(f.readlines()):
            if index == which:
                return line.split('"')[1]
    print('No model to restore')
    exit()


def update_csv(model, file, variables, epoch, iteration, losses):
    with open(file, 'a') as csv_file:
        if 'loss' in file: 
            header = ['Epoch', 'Iteration']
            header.extend(losses)
            writer = csv.DictWriter(csv_file, fieldnames = header)
            line = dict()
            line['Epoch'] = epoch
            line['Iteration'] = iteration
            for ind, val in enumerate(losses):
                line[val] = variables[ind]
        elif 'filter' in file:
            header = ['Epoch', 'Iteration']
            header.extend([str(v.name.split(':')[0].replace('/', '_')) for v in model.gen_filters])
            header.extend([str(v.name.split(':')[0].replace('/', '_')) for v in model.dis_filters])
            writer = csv.DictWriter(csv_file, fieldnames = header)
            line = dict()
            line['Epoch'] = epoch
            line['Iteration'] = iteration
            for var in variables[0]:
                line[var] = variables[0][var]
            for var in variables[1]:
                line[var] = variables[1][var]
        elif 'jacobian' in file:
            writer = csv.writer(csv_file)
            line = [epoch, iteration]
            line.extend(variables)
        elif 'hessian' in file:
            writer = csv.writer(csv_file)
            line = [epoch, iteration]
            line.extend(variables)
        writer.writerow(line)


def setup_csvs(csvs, model, losses):
    loss_csv, filters_s_csv, jacob_s_csv, hessian_s_csv = csvs

    header = ['Epoch', 'Iteration']
    header.extend(losses)
    with open(loss_csv, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)
        writer.writeheader()

    header = ['Epoch', 'Iteration']
    header.extend([str(v.name.split(':')[0].replace('/', '_')) for v in model.gen_filters])
    header.extend([str(v.name.split(':')[0].replace('/', '_')) for v in model.dis_filters])
    with open(filters_s_csv, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)
        writer.writeheader()

    header = ['Epoch', 'Iteration', 'Jacobian Max Singular', 'Jacobian Min Singular']
    with open(jacob_s_csv, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)

    header = ['Epoch', 'Iteration']
    with open(hessian_s_csv, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)

# Setup output folder.
def setup_output(show_epochs, epochs, data, n_images, z_dim, data_out_path, model_name, restore, save_img):
    os.umask(0o002)
    evaluation_path = os.path.join(data_out_path, 'evaluation')
    checkpoints_path = os.path.join(data_out_path, 'checkpoints')
    checkpoints = os.path.join(checkpoints_path, '%s.ckt' % model_name)
    gen_images_path = os.path.join(data_out_path, 'images')
    gen_images = os.path.join(gen_images_path, 'gen_images.h5')
    latent_images = os.path.join(gen_images_path, 'latent_images.h5')

    loss_csv = os.path.join(data_out_path, 'loss.csv')
    filters_s_csv = os.path.join(data_out_path, 'filter_singular_values.csv')
    jacob_s_csv = os.path.join(data_out_path, 'jacobian_singular_values.csv')
    hessian_s_csv = os.path.join(data_out_path, 'hessian_singular_values.csv')
    
    if os.path.isdir(gen_images_path):
         shutil.rmtree(gen_images_path)
    if os.path.isdir(evaluation_path):
         shutil.rmtree(evaluation_path)
    os.makedirs(gen_images_path)
    os.makedirs(evaluation_path)
    
    if not restore:
        if os.path.isdir(checkpoints_path):
            shutil.rmtree(checkpoints_path)
        os.makedirs(checkpoints_path)
    

    image_height = data.training.patch_h
    image_width = data.training.patch_w
    image_channels = data.training.n_channels

    if save_img:
        size_img = (epochs*data.training.iterations)//show_epochs+1
        img_db_shape = (size_img, n_images, image_height, image_width, image_channels)
        latent_db_shape = (size_img, n_images, z_dim)
        hdf5_gen = h5py.File(gen_images, mode='w')
        hdf5_latent = h5py.File(latent_images, mode='w')
        img_storage = hdf5_gen.create_dataset(name='generated_img', shape=img_db_shape, dtype=np.float32)
        latent_storage = hdf5_latent.create_dataset(name='generated_img', shape=latent_db_shape, dtype=np.float32)
    else: 
        img_storage = None
        latent_storage = None

    return img_storage, latent_storage, checkpoints, [loss_csv, filters_s_csv, jacob_s_csv, hessian_s_csv]


# Run session to generate output samples.
def show_generated(session, z_input, z_dim, output_fake, n_images, label_input=None, labels=None, c_input=None, c_dim=None, dim=20, show=True):
    gen_samples = list()
    sample_z = list()
    batch_sample = 20
    for x in range(n_images):
        rand_sample = random.randint(0,batch_sample-1)
        
        z_batch = np.random.uniform(low=-1., high=1., size=(batch_sample, z_dim))
        feed_dict = {z_input:z_batch}
        if c_input is not None:
            c_batch = np.random.normal(loc=0.0, scale=1.0, size=(batch_sample, c_dim))
            feed_dict[c_input] = c_batch
        elif label_input is not None:
            feed_dict[label_input] = labels[:batch_sample, :]
        gen_batch = session.run(output_fake, feed_dict=feed_dict)
        gen_samples.append(gen_batch[rand_sample, :, :, :])
        sample_z.append(z_batch[rand_sample, :])
    if show:
        plot_images(plt_num=n_images, images=np.array(gen_samples), dim=dim)    
    return np.array(gen_samples), np.array(sample_z)


# Method to report parameter in the run.
def report_parameters(model, epochs, restore, data_out_path):
    with open('%s/run_parameters.txt' % data_out_path, 'w') as f:
        f.write('Epochs: %s\n' % (epochs))
        f.write('Restore: %s\n' % (restore))
        for attr, value in model.__dict__.items():
            f.write('%s: %s\n' % (attr, value))


def gather_filters():
    gen_filters = list()
    dis_filters = list()
    for v in tf.trainable_variables():
        if 'filter' in v.name:
            if 'generator' in v.name:
                gen_filters.append(v)
            elif 'discriminator' in v.name:
                dis_filters.append(v)
            elif 'encoder' in v.name:
                dis_filters.append(v)
            else:
                print('No contemplated filter: ', v.name)
                print('Review gather_filters()')
    return gen_filters, dis_filters


def retrieve_csv_data(csv_file, limit_head=2, limit_row=None, sing=0):
    dictionary = dict()
    with open(csv_file) as csvfile:
        reader = csv.DictReader(csvfile)
        for field in reader.fieldnames:
            dictionary[field] = list()
        ind = 0
        for row in reader:
            ind += 1
            if ind < limit_head:
                continue
            elif limit_row is not None and ind >= limit_row:
                break
            for field in reader.fieldnames:
                value = row[field].replace('[', '')
                value = value.replace(']', '')
                if ' ' in value and 'j' in value:
                    value = value.replace('j ', 'j_')
                    value = value.replace(' ', '')
                    value = value.replace('j_', 'j ')
                    value = [complex(val).real for val in value.split(' ')]
                    if sing is None:
                        value = value[0]/value[1]
                    else:
                        value = value[sing]
                elif 'j' in value:
                    value = complex(value)
                    if value.imag > 1e-4:
                        print('[Warning] Imaginary part of singular value larget than 1e-4:', value)
                    value = value.real
                    if value == 0.0:
                        print('[Warning] Min Singular Value Jacobian: [0.0] ', json.dumps(row))
                        value = float(1e-3)
                elif value == '':
                    print('[Warning] Min Singular Value Jacobian: [None]', json.dumps(row))
                    value = float(1e-3)
                else:
                    value = float(value)
                dictionary[field].append(value)

    if 'jacobian' in  csv_file:
        dictionary['Ratio Max/Min'] = list()
        for p in [i for i in range(len(dictionary['Iteration']))]:
            dictionary['Ratio Max/Min'].append(np.log(dictionary['Jacobian Max Singular'][p]/dictionary['Jacobian Min Singular'][p]))

    return dictionary


def plot_data(data1, data2=None, filter1=[], filter2=[], dim=20, total_axis=20, same=False):
    mpl.rcParams['figure.figsize'] = dim, dim
    exclude_b = ['Epoch', 'Iteration']
    fig, ax1 = plt.subplots()
    points = [i for i in range(len(data1['data']['Iteration']))]

    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, 8)]
    random.shuffle(colors)
    ind = 0

    # First data plot
    exclude1 = list()
    exclude1.extend(exclude_b)
    exclude1.extend(filter1)
    ax1.set_xlabel('Iterations (Batch size)')
    ax1.set_ylabel(data1['name'])
    # ax1.set_color_cycle(['red', 'black', 'yellow'])
    for field in data1['data']:
        flag = False
        for exclude in exclude1:
            if exclude in field:
                flag=True
                break
        if flag: continue
        ax1.plot(points, data1['data'][field], label='%s %s' %(data1['name'].split(' ')[1],field), color=colors[ind])
        ind += 1

    every = int(len(points)/total_axis)
    if every == 0: every =1
    plt.xticks(points[0::every], data1['data']['Iteration'][0::every], rotation=45)
    plt.legend(loc='upper left')

    if data2 is not None:
        # Second data plot
        exclude2 = list()
        exclude2.extend(exclude_b)
        exclude2.extend(filter2)
        if not same:   
            ax2 = ax1.twinx()  
            ax2.set_ylabel(data2['name']) 
            plot = ax2
        else:
            plot = ax1
        for field in data2['data']:
            flag = False
            for exclude in exclude2:
                if exclude in field:
                    flag=True
                    break
            if flag: continue
            plot.plot(points, data2['data'][field], label='%s %s' %(data2['name'].split(' ')[1],field), color=colors[ind])
            ind += 1
        plt.xticks(points[0::every], data2['data']['Iteration'][0::every], rotation=45)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.legend(loc='upper right')
    plt.show()

def display_activations(layer_activations, image, images_row, dim=None):
    if dim is not None:
        import matplotlib as mpl
        mpl.rcParams['figure.figsize'] = dim, dim
    num_channels = layer_activations.shape[-1]
    img_width = layer_activations.shape[2]
    img_height = layer_activations.shape[1]
    rows = math.ceil(num_channels/images_row)
    grid = np.zeros((img_height*rows, img_width*images_row))
    
    print('Number of Channels:', num_channels)
    print('Number of Rows:', rows)
    for channel in range(num_channels):
        channel_image = layer_activations[image, :, :, channel]
        channel_image -= channel_image.mean() 
        channel_image /= channel_image.std()
        channel_image *= 64
        channel_image += 128
        channel_image = np.clip(channel_image, 0, 255).astype('uint8')
        grid_row = int(channel/images_row)
        grid_col = channel%images_row
        grid[grid_row*img_height : grid_row*img_height + img_height, grid_col*img_width: grid_col*img_width + img_width] = channel_image

    scale = 1. / num_channels
    plt.figure(figsize=(scale * grid.shape[1], scale * grid.shape[0]))
    plt.matshow(grid)
    
