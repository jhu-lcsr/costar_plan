'''
Splits dataset into train, validation, and test sets.
Inherits existing validation and test sets.
New files are added into training set.

flags:
path to dataset
train file name (optional)
val file name (optional)
test file name (optional)
output name
success_only
ignore_failure
ignore_success
ignore_error
'''
import argparse
import os


def _parse_args():
    parser = argparse.ArgumentParser(description=
        'Splits dataset into train, validation and test sets.'
        'Inherits existing validation and test sets.'
        'New files are added into training set.'
        'If no pre-existing sets of files are indicated, randomize and split the files'
        ' in the folder 8:1:1 for train/val/test.')
    parser.add_argument("--path", type=str,
                        default=os.path.join(os.path.expanduser("~"), '.costar', 'data'),
                        help='path to dataset folder containing many files')
    # parser.add_argument("--ignore_failure", action='store_true',
    #                     default=False, help='skip grasp failure cases')
    # parser.add_argument("--ignore_success", action='store_true',
    #                     default=False, help='skip grasp success cases')
    # parser.add_argument("--ignore_error", action='store_true', default=False,
    #                     help='skip attempts that are both failures and contain errors')
    # parser.add_argument("--success_only", action='store_true', default=False,
    #                     help='only visit stacking data labeled as successful')
    parser.add_argument("--plush", action='store_true', default=False,
                        help='processing plush attempts')
    parser.add_argument("--train", type=str, default='',
                        help='pre-existing filenames for training. '
                        'the file is expected to be in argument `path`')
    parser.add_argument("--val", type=str, default='',
                        help='pre-existing filenames for validation. '
                        'the file is expected to be in argument `path`')
    parser.add_argument("--test", type=str, default='',
                        help='pre-existing filenames for testing. '
                        'the file is expected to be in argument `path`')
    parser.add_argument("--output_name", type=str,
                        default='costar_block_stacking_dataset', help='output file name')
    return vars(parser.parse_args())


def extract_filename_from_url(url):
    filename = url[url.rfind("/")+1:]
    return filename


def get_existing_filenames(path_to_file):
    """Open the file indicated by the input, and output a list of the filenames in the file.

    """
    f = open(path_to_file, 'r')

    filenames = []  # A list to store the filenames in the file
    for line in f:
        line = line.replace('\n', '')  # Remove newline character
        # Extract the file names and add them to the returning list
        filename = extract_filename_from_url(line)
        if not filename:
            print("Empty line extracted.")
            pause()
            continue
        filenames.append(filename)

    f.close()

    print('Read ' + str(len(filenames)) + ' filenames from ' + path_to_file)
    pause()  # DEBUG
    return filenames


def split_dataset(filenames, train_set, val_set, test_set):
    """Split the input filenames into three sets.
    If val_set and test_set are both zero, split the input 8:1:1
    If val_set and test_set have unequal length, match the two lengths
    Add additional files not in val or test sets into training set

    """
    if len(val_set) is 0 and len(test_set) is 0:
        # from math import floor

        # total_samples = len(filenames)
        # ten_percent_samples = int(floor(total_samples / 10))
        ten_percent_samples = 64

        not_train_set = [filename for filename in filenames if filename not in train_set]

        val_set = not_train_set[0:ten_percent_samples]
        test_set = not_train_set[ten_percent_samples:2*ten_percent_samples]
        train_set += not_train_set[2*ten_percent_samples:]

    else:
        """
        for filename in filenames:
            if (filename in train_set) \
                or (filename in val_set) \
                or (filename in test_set):
                continue
            else:
                if filename is '':
                    print("A filename is empty!")
                    pause()
                    continue

                # Add the filenames to val and test until they are of equal length
                if len(val_set) < len(test_set):
                    val_set.append(filename)
                elif len(val_set) > len(test_set):
                    test_set.append(filename)

                # Dump the rest into training set
                else:
                    train_set.append(filename)
        """

        # Select filenames not in test or val set
        not_val_or_test_set = \
            [filename for filename in filenames if
             filename not in val_set and filename not in test_set]

        # Check that val and test set are of the same size
        val_test_len_diff = len(val_set) - len(test_set)

        if val_test_len_diff == 0:
            # val and test set same size => everything belongs to train set
            train_set = not_val_or_test_set
        else:
            # add filenames to val and test until they are of equal length
            if val_test_len_diff > 0:
                # val set is longer
                test_set.extend(not_val_or_test_set[0:val_test_len_diff])
                train_set = not_val_or_test_set[val_test_len_diff:]
            else:
                # test set is longer
                val_set.extend(not_val_or_test_set[0:-val_test_len_diff])
                train_set = not_val_or_test_set[-val_test_len_diff:]

    return train_set, val_set, test_set


def output_file(path, plush, output_prefix, set_name, filenames):
    output_filename = output_prefix + '_' + set_name + '_files.txt'
    print('Writing ' + path + output_filename)

    f = open(os.path.join(path, output_filename), 'w')
    print(f)

    if plush:
        folder = 'blocks_with_plush_toy/'
    else:
        folder = 'blocks_only/'

    prefix_path = \
        "~/.keras/datasets/costar_block_stacking_dataset_v0.4/" + folder

    for filename in filenames:
        # print filename
        f.write(prefix_path + filename + '\n')

    f.close()


def pause():
    _ = input("Press <Enter> to continue...")


def compare_filenames(path, name1, name2):
    """ Quick function meant to check if filenames are the same
    Intended to be used in REPL only
    from split_dataset import compare_filenames

    costar_block_stacking_v0.3_success_only_train_files.txt
    """

    path = os.path.expanduser(path)
    if os.path.isdir(path):
        filenames = os.listdir(path)
    else:
        raise ValueError('Path entered is not a path: ' + path)
    print('Read ' + str(len(filenames)) + ' filenames in the folder')

    # Read files that are success, for now
    filenames = [filename for filename in filenames if '.success.h5f' in filename]
    print('Selecting ' + str(len(filenames)) + ' success files')
    pause()

    file1 = get_existing_filenames(os.path.join(path, name1))
    file2 = get_existing_filenames(os.path.join(path, name2))

    print(name1 + ": " + str(len(file1)) + "; "
          + name2 + ": " + str(len(file2)))

    same = []
    diff = []
    if len(file1) < len(file2):
        file1, file2 = file2, file1

    for filename in file1:
        if filename in file2:
            same.append(filename)
        else:
            diff.append(filename)
    print("same: " + str(len(same)) + "; diff: " + str(len(diff)))

    return same, diff


def main(args, root='root'):
    # Open and get existing training/validation/test files
    # Save the filenames to separate lists
    # Get all the .h5f filenames in the path, compare them to the lists
    # In training, validation, or test set -> continue
    # Not in any set -> add to test set

    # handle no pre-existing files => randomize and split files into 3 sets 8:1:1

    # Get the path to
    path = os.path.expanduser(args['path'])
    if os.path.isdir(path):
        filenames = os.listdir(path)
    else:
        raise ValueError('Path entered is not a path: ' + path)
    print('Read ' + str(len(filenames)) + ' filenames in the folder')

    # Read files that are success, for now
    filenames = [filename for filename in filenames if '.success.h5f' in filename]
    print('Selecting ' + str(len(filenames)) + ' success files')
    args['output_name'] += '_success_only'
    pause()

    # Read filenames for the previous training set
    if not args['train']:
        train_set = []
    else:
        pre_existing_set_file = path + args['train']
        if not os.path.isfile(pre_existing_set_file):
            raise ValueError(
                'Pre-existing training file is not a file: ' +
                pre_existing_set_file)

        train_set = get_existing_filenames(pre_existing_set_file)

    # Read filenames for the previous validation set
    if not args['val']:
        val_set = []
    else:
        pre_existing_set_file = path + args['val']
        if not os.path.isfile(pre_existing_set_file):
            raise ValueError(
                'Pre-existing validating file is not a file: ' +
                pre_existing_set_file)

        val_set = get_existing_filenames(pre_existing_set_file)

    # Read filenames for the previous test set
    if not args['test']:
        test_set = []
    else:
        pre_existing_set_file = path + args['test']
        if not os.path.isfile(pre_existing_set_file):
            raise ValueError(
                'Pre-existing testing file is not a file: ' +
                pre_existing_set_file)

        test_set = get_existing_filenames(pre_existing_set_file)

    # Inform the user that the length of val and test will be matched for output,
    # when the lengths of val and test are not equal
    if len(val_set) is not len(test_set):
        print('Validation set and testing set do not have the same length. '
              'Output results will be adjusted to same size sets')

    # Randomize the filenames
    from random import shuffle
    shuffle(filenames)

    # Split the dataset
    train_set, val_set, test_set = split_dataset(filenames, train_set, val_set, test_set)

    for i in val_set:
        if i in train_set:
            print("val attempt in train set! %s" % i)
            pause()

    for i in test_set:
        if i in train_set:
            print("test attempt in train set! %s" % i)
            pause()

    if (len(train_set) + len(val_set) + len(test_set)) != len(filenames):
        print("ERROR! lenth of train, val and test = %d, %d, %d"
              % (len(train_set), len(val_set), len(test_set)))
        print("Length of all files: %d" % len(filenames))
        pause()
        raise Exception("Something is wrong!")

    # Write the output files
    output_file(path, args['plush'], args['output_name'], 'train', train_set)
    output_file(path, args['plush'], args['output_name'], 'val', val_set)
    output_file(path, args['plush'], args['output_name'], 'test', test_set)


if __name__ == '__main__':
    args = _parse_args()
    main(args)
