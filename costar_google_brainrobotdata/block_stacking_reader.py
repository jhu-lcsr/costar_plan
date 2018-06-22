
import h5py
import os
import io
import glob
from PIL import Image
from skimage.transform import resize

import numpy as np
import json
import keras
from keras.utils import Sequence
from keras.utils import OrderedEnqueuer
import tensorflow as tf


class CostarBlockStackingSequence(Sequence):
    '''Generates a batch of data from the stacking dataset.
    '''
    def __init__(self, list_example_filenames, batch_size=32, shuffle=False, seed=0,
                 resize_shape=None, is_training=True, verbose=0):
        '''Initialization

        #Arguments

        list_Ids: a list of file paths to be read
        batch_size: specifies the size of each batch
        shuffle: boolean to specify shuffle after each epoch
        seed: a random seed to use. If seed is None it will be in order!
        '''
        self.batch_size = batch_size
        self.list_example_filenames = list_example_filenames
        self.shuffle = shuffle
        self.seed = seed
        self.resize_shape = resize_shape
        self.is_training = is_training
        self.verbose = verbose
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        """
        return int(np.floor(len(self.list_example_filenames) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        if self.verbose > 0:
            print("batch getitem indices:" + str(indexes))
        # Find list of example_filenames
        list_example_filenames_temp = [self.list_example_filenames[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_example_filenames_temp)

        return X, y

    def on_epoch_end(self):
        #Updates indexes after each epoch
        if self.seed is not None and not self.is_training:
            # repeat the same order if we're validating or testing
            # continue the large random sequence for training
            np.random.seed(self.seed)
        self.indexes = np.arange(len(self.list_example_filenames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_Ids):
        """ Generates data containing batch_size samples

        # Arguments

        list_Ids: a list of file paths to be read
        """

        def JpegToNumpy(jpeg):
            stream = io.BytesIO(jpeg)
            im = np.asarray(Image.open(stream))
            try:
                return im.astype(np.uint8)
            except(TypeError) as exception:
                print("Failed to convert PIL image type",exception)
                print("type ",type(im),"len ",len(im))

        def ConvertImageListToNumpy(data, format='numpy', data_format='NHWC'):
            """ Convert a list of binary jpeg or png files to numpy format.

            # Arguments

            data: a list of binary jpeg images to convert
            format: default 'numpy' returns a 4d numpy array,
                'list' returns a list of 3d numpy arrays
            """
            length = len(data)
            imgs = []
            for raw in data:
                img = JpegToNumpy(raw)
                if data_format == 'NCHW':
                    img = np.transpose(img, [2, 0, 1])
                imgs.append(img)
            if format == 'numpy':
                imgs = np.array(imgs)
            return imgs
        # Initialization
        if self.verbose > 0:
            print("generating batch: " + str(list_Ids))
        X = []
        init_images = []
        current_images = []
        poses = []
        y = []
        action_labels = []

        # Generate data
        for i, example_filename in enumerate(list_Ids):
            if self.verbose > 0:
                print('reading: ' + str(i) + ' path: ' + str(example_filename))
            # Store sample
            #X[i,] = np.load('data/' + example_filename + '.npy')
            x = ()
        try:
            with h5py.File(example_filename, 'r') as data:
                if 'gripper_action_goal_idx' not in list(data.keys()) or 'gripper_action_label' not in list(data.keys()):
                    raise ValueError('You need to run preprocessing before this will work! see view_convert_dataset.py --preprocess_inplace.')
                #indices = [0]
                # len of goal indexes is the same as the number of images, so this saves loading all the images
                all_goal_ids = np.array(data['gripper_action_goal_idx'])
                if self.seed is not None:
                    image_indices = np.random.randint(1, len(all_goal_ids)-1, 1)
                else:
                    raise NotImplementedError
                indices = [0] + list(image_indices)
                if self.verbose > 0:
                    print("Indices --",indices)
                rgb_images = list(data['image'][indices])
                rgb_images = ConvertImageListToNumpy(np.squeeze(rgb_images), format='numpy')
                #resize using skimage
                rgb_images_resized = []
                for k, images in enumerate(rgb_images):
                    rgb_images_resized.append(resize(images, (224, 224, 3)))
                init_images.append(rgb_images_resized[0])
                current_images.append(rgb_images_resized[1])
                poses.append(np.array(data['pose'][indices[1:]])[0])
                # x = x + tuple([rgb_images[indices]])
                # x = x + tuple([np.array(data['pose'])[indices]])
                for j in indices[1:]:
                    action = np.zeros(41)
                    action[data['gripper_action_label'][j]] = 1
                    action_labels.append(action)
                # action_labels = np.array(action_labels)


                #print(action_labels)
                # x = x + tuple([action_labels])
                #X.append(x)
                # action_labels = np.unique(data['gripper_action_label'])
                # print(np.array(data['labels_to_name']).shape)
                #X.append(np.array(data['pose'])[indices])

                # Store class
                label = ()
                #change to goals computed
                index1 = indices[1]
                goal_ids = all_goal_ids[index1]
                # print(index1)
                label = np.array(data['pose'])[goal_ids]
                #print(type(label))
                # for items in list(data['all_tf2_frames_from_base_link_vec_quat_xyzxyzw_json'][indices]):
                #     json_data = json.loads(items.decode('UTF-8'))
                #     label = label + tuple([json_data['gripper_center']])
                #     print(np.array(json_data['gripper_center']))
                    #print(json_data.keys())
                    #y.append(np.array(json_data['camera_rgb_frame']))
                y.append(label)
        except IOError as ex:
            print('Error: Skipping file due to IO error when opening ' +
                  example_filename + ': ' + str(ex) + ' using the last example twice for batch')

        action_labels, init_images, current_images, poses = np.array(action_labels), np.array(init_images), np.array(current_images), np.array(poses)
        action_poses_vec = np.concatenate([poses, action_labels], axis=1)
        # print("---",init_images.shape)
        # init_images = tf.image.resize_images(init_images,[224,224])
        # current_images = tf.image.resize_images(current_images,[224,224])
        # print("---",init_images.shape)
        # X = init_images
        X = [init_images, current_images, action_poses_vec]
        # print("type=======",type(X))
        # print("shape=====",X.shape)
        y = np.array(y)
        batch = (X, y)

        if self.verbose > 0:
            print("generated batch: " + str(list_Ids))

        return batch


def block_stacking_generator(sequence):

    # training_generator = CostarBlockStackingSequence(filenames, batch_size=1)
    epoch_size = len(sequence)
    step = 0
    while True:
        if step > epoch_size:
            step = 0
            sequence.on_epoch_end()
        batch = sequence.__getitem__(step)
        step += 1
        yield batch

if __name__ == "__main__":
    tf.enable_eager_execution()
    filenames = glob.glob(os.path.expanduser('~/.keras/datasets/costar_task_planning_stacking_dataset_v0.1'))
    #print(filenames)
    training_generator = CostarBlockStackingSequence(filenames, batch_size=1, verbose = 0)
    num_batches = len(training_generator)

    bsg = block_stacking_generator(training_generator)
    iter(bsg)
    from tqdm import tqdm as tqdm
    progress = tqdm(range(num_batches))
    for i in progress:
        data = next(bsg)
        progress.set_description('step: ' + str(i) + ' data type: ' + str(type(data)))
    # a = next(training_generator)
    enqueuer = OrderedEnqueuer(
                    training_generator,
                    use_multiprocessing=False,
                    shuffle=True)
    enqueuer.start(workers=1, max_queue_size=1)
    generator = iter(enqueuer.get())
    print("-------------------")
    generator_ouput = next(generator)
    print("-------------------op")
    x, y = generator_ouput
    # print(x.shape)
    # print(y.shape)

    # X,y=training_generator.__getitem__(1)
    #print(X.keys())
    # print(X[0].shape)
    # print(X[0].shape)
    # print(y[0])
