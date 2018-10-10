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
    parser.add_argument("--success_only", action='store_true', default=False,
                        help='only visit stacking data labeled as successful')
    parser.add_argument("--split_all", action='store_true', default=False,
                        help='Split all datasets into success, failure, and error sets. '
                             'Requires train/val/test from success_only subset')
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
    # pause()  # DEBUG
    return filenames


def split_dataset(filenames, train_set, val_set, test_set, val_len=64, test_len=64):
    """Split the input filenames into three sets.
    If val_set and test_set are empty, the sets will be of length val_len and test_len.
    If val_set and test_set have unequal length, match the two lengths
    Add additional files not in val or test sets into training set
    """
    if len(val_set) is 0 and len(test_set) is 0:
        if len(train_set) != 0:
            not_train_set = [filename for filename in filenames
                             if filename not in train_set]
        else:
            not_train_set = filenames

        val_set = not_train_set[0:val_len]
        test_set = not_train_set[val_len:val_len+test_len]
        train_set += not_train_set[val_len+test_len:]

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
    # print(f)

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


def split_success_only(args, filenames, path):
    # Read files that are success
    filenames = [filename for filename in filenames if '.success.h5f' in filename]
    print('Selecting ' + str(len(filenames)) + ' success files')
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
    output_file(path, args['plush'], args['output_name'], 'success_only_train', train_set)
    output_file(path, args['plush'], args['output_name'], 'success_only_val', val_set)
    output_file(path, args['plush'], args['output_name'], 'success_only_test', test_set)


def split_all(args, filenames, path):
    # Get the success, failure, and error filenames with nonzero frames
    success_filenames, failure_filenames, error_filenames = count_nonzero_files(path, filenames)
    pause()  # DEBUG

    # Calculate the percentage of success, failure and error
    total_file_count = len(success_filenames) + len(failure_filenames) + len(error_filenames)
    success_ratio = len(success_filenames) / total_file_count
    failure_ratio = len(failure_filenames) / total_file_count
    error_ratio = len(error_filenames) / total_file_count
    print("Total: %d files" % total_file_count)
    print("Ratios: {:.2f}% success, {:.2f}% failure(no error), {:.2f}% error".format(
            success_ratio*100, failure_ratio*100, error_ratio*100))
    pause()  # DEBUG

    # Read the train/val set from success_only subset
    if args['plush']:
        default_name = 'costar_plush_block_stacking_v0.4_success_only_'
    else:
        default_name = 'costar_block_stacking_v0.4_success_only_'
    # Read filenames for the previous training set
    if not args['train']:
        # Look for v0.4 success only train filenames
        print('No train file is specified. Trying to open v0.4 success only...')
        pre_existing_set_file = path + default_name + 'train_files.txt'
    else:
        pre_existing_set_file = path + args['train']

    if not os.path.isfile(pre_existing_set_file):
        raise ValueError(
            'Pre-existing training file is not a file: ' +
            pre_existing_set_file)
    success_train_len = len(get_existing_filenames(pre_existing_set_file))

    # Read filenames for the previous validation set
    if not args['val']:
        # Look for v0.4 success only val filenames
        print('No val file is specified. Trying to open v0.4 success only...')
        pre_existing_set_file = path + default_name + 'val_files.txt'
    else:
        pre_existing_set_file = path + args['val']

    if not os.path.isfile(pre_existing_set_file):
        raise ValueError(
            'Pre-existing validating file is not a file: ' +
            pre_existing_set_file)
    success_val_len = len(get_existing_filenames(pre_existing_set_file))

    # Read filenames for the previous test set
    if not args['test']:
        # Look for v0.4 success only train filenames
        print('No test file is specified. Trying to open v0.4 success only...')
        pre_existing_set_file = path + default_name + 'test_files.txt'
    else:
        pre_existing_set_file = path + args['test']

    if not os.path.isfile(pre_existing_set_file):
        raise ValueError(
            'Pre-existing testing file is not a file: ' +
            pre_existing_set_file)
    success_test_len = len(get_existing_filenames(pre_existing_set_file))

    # Calculate set size for failure and error, based on success_only subset
    multiplier_failure = len(failure_filenames)/len(success_filenames)
    failure_val_len = int(round(success_val_len*multiplier_failure))
    failure_test_len = int(round(success_test_len*multiplier_failure))
    failure_train_len = len(failure_filenames) - (failure_val_len + failure_test_len)
    multiplier_error = len(error_filenames)/len(success_filenames)
    error_val_len = int(round(success_val_len*multiplier_error))
    error_test_len = int(round(success_test_len*multiplier_error))
    error_train_len = len(error_filenames) - (error_val_len + error_test_len)
    print("Successfully read success_only filenames: {0} train, {1} val, {2} test".format(
            success_train_len, success_val_len, success_test_len))
    print("Length for failure sets: {0} train, {1} val, {2} test".format(
            failure_train_len + error_train_len, 
            failure_val_len + error_val_len, 
            failure_test_len + error_test_len))
    print("Length for failure (no error) sets: {0} train, {1} val, {2} test".format(
            failure_train_len, failure_val_len, failure_test_len))
    print("Length for error sets: {0} train, {1} val, {2} test".format(
            error_train_len, error_val_len, error_test_len))
    pause()
    
    # Randomize the filenames
    from random import shuffle
    shuffle(failure_filenames)
    shuffle(error_filenames)

    # Split the dataset for failure and error
    fail_train_set, fail_val_set, fail_test_set = \
        split_dataset(failure_filenames, [], [], [], failure_val_len, failure_test_len)
    err_train_set,  err_val_set,  err_test_set = \
        split_dataset(error_filenames, [], [], [], error_val_len, error_test_len)

    for i in fail_val_set:
        if i in fail_train_set:
            print("fail: val attempt in train set! %s" % i)
            pause()
    for i in fail_test_set:
        if i in fail_train_set:
            print("fail: test attempt in train set! %s" % i)
            pause()

    for i in err_val_set:
        if i in err_train_set:
            print("err: val attempt in train set! %s" % i)
            pause()

    for i in err_test_set:
        if i in err_train_set:
            print("err: test attempt in train set! %s" % i)
            pause()
    pause()

    # Write the output files
    output_file(path, args['plush'], args['output_name'], 'task_failure_only_train', fail_train_set)
    output_file(path, args['plush'], args['output_name'], 'task_failure_only_val', fail_val_set)
    output_file(path, args['plush'], args['output_name'], 'task_failure_only_test', fail_test_set)
    output_file(path, args['plush'], args['output_name'], 'error_failure_only_train', err_train_set)
    output_file(path, args['plush'], args['output_name'], 'error_failure_only_val', err_val_set)
    output_file(path, args['plush'], args['output_name'], 'error_failure_only_test', err_test_set)

    # Error is also a type of failure!
    fail_train_set += err_train_set
    fail_val_set += err_val_set
    fail_test_set += err_test_set
    output_file(path, args['plush'], args['output_name'], 'all_failure_only_train', fail_train_set)
    output_file(path, args['plush'], args['output_name'], 'all_failure_only_val', fail_val_set)
    output_file(path, args['plush'], args['output_name'], 'all_failure_only_test', fail_test_set)


def count_nonzero_files(path, filenames):
    '''
    Open the files and check frame count. Skip files with 0 frame.

    :param filenames: .h5f filenames in the folder
    :return: Lists of success/failure/error filenames with nonzero frames
    '''
    import h5py  # Needs h5py to open the files and check frame count
    # TODO: Write total frames into csv file as a new column

    # Open the files to check frame count. Skip files with 0 frame.
    error_filenames = []
    failure_filenames = []
    success_filenames = []
    skip_count = 0
    i = 0
    for filename in filenames:
        # print(filename)
        i += 1
        try:
            with h5py.File(os.path.join(path, filename), 'r') as data:
                try:
                    total_frames = len(data['image'])
                except KeyError as e:
                    print('Skipping %s for KeyError' % filename)
                    continue

                if total_frames == 0:  # Skip files with 0 frame
                    # print('Skipping %s since it has 0 image frame' % filename)
                    skip_count += 1
                    continue

                if 'error' in filename:
                    error_filenames += [filename]
                elif 'failure' in filename:
                    failure_filenames += [filename]
                else:  # success
                    success_filenames += [filename]
                # elif 'success' in filename:
                #     success_filenames += [filename]
                # else:  # BUG: Sanity check for debugging
                #     raise Exception('Somthing is wrong!')
        except IOError as ex:
            print('Skipping %s for IO error' % filename)

    print("Counted {:d} success files, {:d} failure files, and {:d} error files.".format(
            len(success_filenames), len(failure_filenames), len(error_filenames)))
    print("Skipped %d files since they have 0 image frame" % skip_count)

    return success_filenames, failure_filenames, error_filenames


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

    filenames = [filename for filename in filenames if '.h5f' in filename]
    print('Read ' + str(len(filenames)) + ' h5f filenames in the folder')

    if args['success_only'] and args['split_all']:
        raise ValueError('success_only and split_all are mutually exclusive. Please choose just one.')
    elif args['success_only']:
        split_success_only(args, filenames, path)
    elif args['split_all']:
        split_all(args, filenames, path)


if __name__ == '__main__':
    args = _parse_args()
    main(args)
