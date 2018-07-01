
import h5py
import os
import io
import sys
import glob
import traceback
from PIL import Image
from skimage.transform import resize

import numpy as np
import json
import keras
from keras.utils import Sequence
from keras.utils import OrderedEnqueuer
import tensorflow as tf
import grasp_metrics
import keras_applications
import keras_preprocessing


def random_eraser(input_img, p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=True):
    """ Cutout and random erasing algorithms for data augmentation

    source:
    https://github.com/yu4u/cutout-random-erasing/blob/master/random_eraser.py
    """
    img_h, img_w, img_c = input_img.shape
    p_1 = np.random.rand()

    if p_1 > p:
        return input_img

    while True:
        s = np.random.uniform(s_l, s_h) * img_h * img_w
        r = np.random.uniform(r_1, r_2)
        w = int(np.sqrt(s / r))
        h = int(np.sqrt(s * r))
        left = np.random.randint(0, img_w)
        top = np.random.randint(0, img_h)

        if left + w <= img_w and top + h <= img_h:
            break

    if pixel_level:
        c = np.random.uniform(v_l, v_h, (h, w, img_c))
    else:
        c = np.random.uniform(v_l, v_h)

    input_img[top:top + h, left:left + w, :] = c

    return input_img


def tile_vector_as_image_channels_np(vector_op, image_shape):
    """
    Takes a vector of length n and an image shape BHWC,
    and repeat the vector as channels at each pixel.

    # Params

      vector_op: A tensor vector to tile.
      image_shape: A list of integers [width, height] with the desired dimensions.
    """
    ivs = np.shape(vector_op)
    # reshape the vector into a single pixel
    vector_pixel_shape = [ivs[0], 1, 1, ivs[1]]
    vector_op = np.reshape(vector_op, vector_pixel_shape)
    # tile the pixel into a full image
    tile_dimensions = [1, image_shape[1], image_shape[2], 1]
    vector_op = np.tile(vector_op, tile_dimensions)
    # if K.backend() is 'tensorflow':
    #     output_shape = [ivs[0], image_shape[1], image_shape[2], ivs[1]]
    #     vector_op.set_shape(output_shape)
    return vector_op


def concat_images_with_tiled_vector_np(images, vector):
    """Combine a set of images with a vector, tiling the vector at each pixel in the images and concatenating on the channel axis.

    # Params

        images: list of images with the same dimensions
        vector: vector to tile on each image. If you have
            more than one vector, simply concatenate them
            all before calling this function.

    # Returns

    """
    if not isinstance(images, list):
        images = [images]
    image_shape = np.shape(images[0])
    tiled_vector = tile_vector_as_image_channels_np(vector, image_shape)
    images.append(tiled_vector)
    combined = np.concatenate(images, axis=-1)

    return combined


class CostarBlockStackingSequence(Sequence):
    '''Generates a batch of data from the stacking dataset.

    # TODO(ahundt) match the preprocessing /augmentation apis of cornell & google dataset
    '''
    def __init__(self, list_example_filenames,
                 label_features_to_extract=None, data_features_to_extract=None,
                 total_actions_available=41,
                 batch_size=32, shuffle=False, seed=0,
                 is_training=True, random_augmentation=None, output_shape=None, verbose=0):
        '''Initialization

        #Arguments

        list_Ids: a list of file paths to be read
        batch_size: specifies the size of each batch
        shuffle: boolean to specify shuffle after each epoch
        seed: a random seed to use. If seed is None it will be in order!
        # TODO(ahundt) better notes about the two parameters below. See choose_features_and_metrics() in cornell_grasp_trin.py.
        label_features_to_extract: defaults to regression options, classification options are also available
        data_features_to_extract: defaults to regression options, classification options are also available
            Options include 'image_0_image_n_vec_xyz_aaxyz_nsc_15' which is a giant NHWC cube of image and pose data,
            'current_xyz_aaxyz_nsc_8', 'proposed_goal_xyz_aaxyz_nsc_8'.
        random_augmentation: None or a float value between 0 and 1 indiciating how frequently random augmentation should be applied.
        '''
        self.batch_size = batch_size
        self.list_example_filenames = list_example_filenames
        self.shuffle = shuffle
        self.seed = seed
        self.output_shape = output_shape
        self.is_training = is_training
        self.verbose = verbose
        self.on_epoch_end()
        self.output_shape = output_shape
        self.label_features_to_extract = label_features_to_extract
        # TODO(ahundt) total_actions_available can probably be extracted from the example hdf5 files and doesn't need to be a param
        self.data_features_to_extract = data_features_to_extract
        self.total_actions_available = total_actions_available
        self.random_augmentation = random_augmentation
        # if crop_shape is None:
        #     # height width 3
        #     crop_shape = (224, 224, 3)
        # self.crop_shape = crop_shape

    def __len__(self):
        """Denotes the number of batches per epoch
        """
        return int(np.floor(len(self.list_example_filenames) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        """
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
        """ Updates indexes after each epoch
        """
        if self.seed is not None and not self.is_training:
            # repeat the same order if we're validating or testing
            # continue the large random sequence for training
            np.random.seed(self.seed)
        self.indexes = np.arange(len(self.list_example_filenames))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def encode_pose(self, pose):
        """ Encode n x 7 array of poses to something that might be regressed by 7 sigmoid values
        """
        # change unit quaternion to be in range [0, 1]
        # change x y z to be in what should hopefully be [0, 1]
        xyz = (pose[:, :3] / 5) + 0.5
        quat = (pose[:, 3:] / 2) + 0.5
        if self.verbose > 0:
            print('encode xyz: ' + str(xyz) + '\n encode quat: ' + str(quat))
        encoded_pose = np.concatenate([xyz, quat], axis=-1)
        return encoded_pose

    def decode_pose(self, pose):
        """ Decode n x 7 array of poses encoding in encode_pose
        """
        # change unit quaternion to be in range [-1, 1]
        # change x y z to be in what should hopefully be [-2.5, 2.5]
        xyz = (pose[:, :3] - 0.5) * 5
        quat = (pose[:, 3:] - 0.5) * 2
        if self.verbose > 0:
            print('decode xyz: ' + str(xyz) + '\n decode quat: ' + str(quat))
        encoded_pose = np.concatenate([xyz, quat], axis=-1)
        return encoded_pose

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
                print("Failed to convert PIL image type", exception)
                print("type ", type(im), "len ", len(im))

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
        try:
            # Initialization
            if self.verbose > 0:
                print("generating batch: " + str(list_Ids))
            X = []
            init_images = []
            current_images = []
            poses = []
            y = []
            action_labels = []
            action_successes = []
            example_filename = ''
            if isinstance(list_Ids, int):
                # if it is just a single int
                # make it a list so we can iterate
                list_Ids = [list_Ids]

            # Generate data
            for i, example_filename in enumerate(list_Ids):
                if self.verbose > 0:
                    print('reading: ' + str(i) + ' path: ' + str(example_filename))
                # Store sample
                # X[i,] = np.load('data/' + example_filename + '.npy')
                x = ()
                try:
                    with h5py.File(example_filename, 'r') as data:
                        if 'gripper_action_goal_idx' not in data or 'gripper_action_label' not in data:
                            raise ValueError('block_stacking_reader.py: You need to run preprocessing before this will work! \n' +
                                             '    python2 ctp_integration/scripts/view_convert_dataset.py --path ~/.keras/datasets/costar_block_stacking_dataset_v0.2 --preprocess_inplace gripper_action --write'
                                             '\n File with error: ' + str(example_filename))
                        # indices = [0]
                        # len of goal indexes is the same as the number of images, so this saves loading all the images
                        all_goal_ids = np.array(data['gripper_action_goal_idx'])
                        if self.seed is not None:
                            image_indices = np.random.randint(1, len(all_goal_ids)-1, 1)
                        else:
                            raise NotImplementedError
                        indices = [0] + list(image_indices)
                        if self.verbose > 0:
                            print("Indices --", indices)
                        rgb_images = list(data['image'][indices])
                        rgb_images = ConvertImageListToNumpy(np.squeeze(rgb_images), format='numpy')
                        # resize using skimage
                        rgb_images_resized = []
                        for k, images in enumerate(rgb_images):
                            if (self.is_training and self.random_augmentation is not None and
                                    np.random.random() > self.random_augmentation):
                                # apply random shift to the images before resizing
                                images = keras_preprocessing.image.random_shift(
                                    images,
                                    # height, width
                                    1./(48. * 2.), 1./(64. * 2.),
                                    row_axis=0, col_axis=1, channel_axis=2)
                            # TODO(ahundt) improve crop/resize to match cornell_grasp_dataset_reader
                            if self.output_shape is not None:
                                resized_image = resize(images, self.output_shape)
                            else:
                                resized_image = images
                            if self.is_training:
                                # do some image augmentation with random erasing & cutout
                                resized_image = random_eraser(resized_image)
                            rgb_images_resized.append(resized_image)

                        init_images.append(rgb_images_resized[0])
                        current_images.append(rgb_images_resized[1])
                        poses.append(np.array(data['pose'][indices[1:]])[0])
                        # x = x + tuple([rgb_images[indices]])
                        # x = x + tuple([np.array(data['pose'])[indices]])

                        if (self.data_features_to_extract is not None and
                                'image_0_image_n_vec_xyz_aaxyz_nsc_15' in self.data_features_to_extract):
                            # normalized floating point encoding of action vector
                            # from 0 to 1 in a single float which still becomes
                            # a 2d array of dimension batch_size x 1
                            # np.expand_dims(data['gripper_action_label'][indices[1:]], axis=-1) / self.total_actions_available
                            for j in indices[1:]:
                                action = [float(data['gripper_action_label'][j] / self.total_actions_available)]
                                action_labels.append(action)
                        else:
                            # one hot encoding
                            for j in indices[1:]:
                                # generate the action label one-hot encoding
                                action = np.zeros(self.total_actions_available)
                                action[data['gripper_action_label'][j]] = 1
                                action_labels.append(action)
                        # action_labels = np.array(action_labels)

                        # print(action_labels)
                        # x = x + tuple([action_labels])
                        # X.append(x)
                        # action_labels = np.unique(data['gripper_action_label'])
                        # print(np.array(data['labels_to_name']).shape)
                        # X.append(np.array(data['pose'])[indices])

                        # Store class
                        label = ()
                        # change to goals computed
                        index1 = indices[1]
                        goal_ids = all_goal_ids[index1]
                        # print(index1)
                        label = np.array(data['pose'])[goal_ids]
                        # print(type(label))
                        # for items in list(data['all_tf2_frames_from_base_link_vec_quat_xyzxyzw_json'][indices]):
                        #     json_data = json.loads(items.decode('UTF-8'))
                        #     label = label + tuple([json_data['gripper_center']])
                        #     print(np.array(json_data['gripper_center']))
                            # print(json_data.keys())
                            # y.append(np.array(json_data['camera_rgb_frame']))
                        y.append(label)
                        if 'success' in example_filename:
                            action_successes = action_successes + [1]
                        else:
                            action_successes = action_successes + [0]
                except IOError as ex:
                    print('Error: Skipping file due to IO error when opening ' +
                          example_filename + ': ' + str(ex) + ' using the last example twice for batch')

            action_labels = np.array(action_labels)
            init_images = keras_applications.imagenet_utils._preprocess_numpy_input(
                np.array(init_images, dtype=np.float32),
                data_format='channels_last', mode='tf')
            current_images = keras_applications.imagenet_utils._preprocess_numpy_input(
                np.array(current_images, dtype=np.float32),
                data_format='channels_last', mode='tf')
            poses = np.array(poses)

            # print('poses shape: ' + str(poses.shape))
            encoded_poses = grasp_metrics.batch_encode_xyz_qxyzw_to_xyz_aaxyz_nsc(
                poses, random_augmentation=self.is_training)

            epsilon = 1e-3
            if np.any(encoded_poses < 0 - epsilon) or np.any(encoded_poses > 1 + epsilon):
                raise ValueError('An encoded pose was outside the [0,1] range! Update your encoding. poses: ' +
                                 str(poses) + ' encoded poses: ' + str(encoded_poses))
            # print('encoded poses shape: ' + str(encoded_poses.shape))
            # print('action labels shape: ' + str(action_labels.shape))
            # print('encoded poses vec shape: ' + str(action_poses_vec.shape))
            # print("---",init_images.shape)
            # init_images = tf.image.resize_images(init_images,[224,224])
            # current_images = tf.image.resize_images(current_images,[224,224])
            # print("---",init_images.shape)
            # X = init_images
            if (self.data_features_to_extract is None or
                    'current_xyz_3' in self.data_features_to_extract or
                    'image_0_image_n_vec_xyz_10' in self.data_features_to_extract):
                # default, regression input case
                action_poses_vec = np.concatenate([encoded_poses[:, :3], action_labels], axis=-1)
                X = [init_images, current_images, action_poses_vec]
            elif (self.data_features_to_extract is None or
                    'current_xyz_aaxyz_nsc_8' in self.data_features_to_extract or
                    'image_0_image_n_vec_xyz_aaxyz_nsc_15' in self.data_features_to_extract):
                # default, regression input case
                action_poses_vec = np.concatenate([encoded_poses, action_labels], axis=-1)
                X = [init_images, current_images, action_poses_vec]
            elif 'proposed_goal_xyz_aaxyz_nsc_8' in self.data_features_to_extract:
                # classification input case
                proposed_and_current_action_vec = np.concatenate([encoded_poses, action_labels, y], axis=-1)
                X = [init_images, current_images, proposed_and_current_action_vec]

            else:
                raise ValueError('Unsupported data input: ' + str(self.data_features_to_extract))

            if (self.data_features_to_extract is not None and 'image_0_image_n_vec_xyz_aaxyz_nsc_15' in self.data_features_to_extract):
                # make the giant data cube if it is requested
                X = concat_images_with_tiled_vector_np(X[:2], X[2:])

            # print("type=======",type(X))
            # print("shape=====",X.shape)

            # determine the label
            if self.label_features_to_extract is None or 'grasp_goal_xyz_3' in self.label_features_to_extract:
                # default, regression to translation case, see semantic_translation_regression in cornell_grasp_train.py
                y = grasp_metrics.batch_encode_xyz_qxyzw_to_xyz_aaxyz_nsc(y, random_augmentation=self.random_augmentation)
                y = y[:, :3]
            elif self.label_features_to_extract is None or 'grasp_goal_xyz_aaxyz_nsc_8' in self.label_features_to_extract:
                # default, regression label case
                y = grasp_metrics.batch_encode_xyz_qxyzw_to_xyz_aaxyz_nsc(y, random_augmentation=self.random_augmentation)
            elif 'grasp_success' in self.label_features_to_extract or 'action_success' in self.label_features_to_extract:
                # classification label case
                y = action_successes
            else:
                raise ValueError('Unsupported label: ' + str(action_labels))

            # Debugging checks
            if X is None:
                raise ValueError('Unsupported input data for X: ' + str(x))
            if y is None:
                raise ValueError('Unsupported input data for y: ' + str(x))

            # Assemble the data batch
            batch = (X, y)

            if self.verbose > 0:
                # diff should be nonzero for most timesteps except just before the gripper closes!
                print('encoded current poses: ' + str(poses) + ' labels: ' + str(y) + ' diff: ' + str(poses - y))
                print("generated batch: " + str(list_Ids))
        except Exception as ex:
            print('CostarBlockStackingSequence: Keras will often swallow exceptions without a stack trace, '
                  'so we are printing the stack trace here before re-raising the error.')
            ex_type, ex, tb = sys.exc_info()
            traceback.print_tb(tb)
            # deletion must be explicit to prevent leaks
            # https://stackoverflow.com/a/16946886/99379
            del tb
            raise

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
    filenames = glob.glob(os.path.expanduser('~/.keras/datasets/costar_block_stacking_dataset_v0.2/*success.h5f'))
    # print(filenames)
    training_generator = CostarBlockStackingSequence(
        filenames, batch_size=1, verbose=0,
        label_features_to_extract='grasp_goal_xyz_aaxyz_nsc_8',
        data_features_to_extract=['current_xyz_aaxyz_nsc_8'])
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
