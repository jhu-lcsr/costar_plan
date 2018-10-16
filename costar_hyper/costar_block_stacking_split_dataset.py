'''
Splits dataset into train, validation, and test sets.
Inherits existing validation and test sets. New files are added into training set.

To split the success_only subset or to add new files ot the success_only subset, call:

python costar_block_stacking_split_dataset.py --path /path/to/dataset/folder\
    --success_only (--plush) (--train train/txt/filename)                   \
    (--val val/txt/filename) (-test test/txt/filename/)                     \
    --output_name [filename prefix for the output train/val/test filenames]

To split all dataset, i.e. split error files and failure files into train/val/test sets,
call the following command after success_only subset is splitted:

python costar_block_stacking_split_dataset.py --path /path/to/dataset/folder     \
    --split_all (--plush) --train success_only/train/txt/filename             \
    --val [success_only val txt filename] --test [success_only test txt filename]\
    --output_name [filename prefix for the output train/val/test filenames]

This will output task_failure_only, error_failure_only, and all_failure_only
train/val/test filenames as 9 separate txt files.

Author: Chia-Hung "Rexxar" Lin (rexxarchl)
Apache License 2.0 https://www.apache.org/licenses/LICENSE-2.0
'''
import argparse
import os
import random
try:
    import h5py  # Needs h5py to open the files and check frame count
except ImportError:
    print("h5py is not available.")
    h5py = None


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Splits dataset into train, validation and test sets. '
                    'Inherits existing validation and test sets. '
                    'New files are added into training set. '
                    'If no pre-existing sets of files are indicated, randomize and split '
                    'the files in the folder 8:1:1 for train/val/test.')
    parser.add_argument("--path", type=str,
                        default=os.path.join(
                            os.path.expanduser("~"),
                            '.keras/datasets/costar_block_stacking_dataset_v0.4/'),
                        help='path to dataset folder containing many files')
    parser.add_argument("--dataset_path", type=str, default='/.keras/dataset/',
                        help='The folder that is expected stores the dataset. '
                             'Filenames in the output file will reference this path.')
    parser.add_argument("--dataset_name", type=str,
                        default='costar_block_stacking_dataset',
                        help='Dataset name to store under dataset path.'
                             'Filenames in the output file will reference this name.')
    # parser.add_argument("--dataset_version", type=str, default='v0.4',
    #                     help='The current version of the dataset.')
    parser.add_argument("--success_only", action='store_true', default=False,
                        help='Only visit stacking data labeled as successful')
    # parser.add_argument("--train", type=str, default='',
    #                     help='Pre-existing filenames for training. '
    #                     'the file is expected to be in argument `path`')
    # parser.add_argument("--val", type=str, default='',
    #                     help='Pre-existing filenames for validation. '
    #                     'the file is expected to be in argument `path`')
    # parser.add_argument("--test", type=str, default='',
    #                     help='Pre-existing filenames for testing. '
    #                     'the file is expected to be in argument `path`')
    parser.add_argument("--output_name", type=str,
                        default='costar_block_stacking_dataset', help='output file name')
    parser.add_argument("--val_len", type=int, default=None,
                        help='Expected val set length')
    parser.add_argument("--test_len", type=int, default=None,
                        help='Expected test set length')
    parser.add_argument("--seed", type=int, default=0,
                        help='Random seed for reproducing the output lists')
    parser.add_argument("--write", action='store_true', default=False,
                        help='Write to output files')
    parser.add_argument("--existing_file_prefix", type=str, nargs='+',
                        default=["costar_plush_block_stacking_dataset_v0.4",
                                 "costar_block_stacking_dataset_v0.4"],
                        help="Existing txt file prefixes to look for when opening "
                             "train/val/test files.")
    return vars(parser.parse_args())


def extract_filename_from_url(url):
    '''Extract the string after the last '/' in the input `url`
    '''
    filename = url[url.rfind("/")+1:]
    return filename


def get_existing_filenames(path_to_file):
    '''Open the file indicated by the input, and output a list of the filenames in the file.
    '''
    f = open(path_to_file, 'r')

    filenames = []  # A list to store the filenames in the file
    for line in f:
        line = line.replace('\n', '')  # Remove newline character
        # Extract the file names and add them to the returning list
        filename = extract_filename_from_url(line)
        if not filename:
            print("get_existing_filenames: Empty line extracted.")
            continue
        filenames.append(filename)

    f.close()

    print('Read ' + str(len(filenames)) + ' filenames from ' + path_to_file)
    return filenames


def output_file(dataset_path, dataset_name, dir_path, dir_name,
                category_name, subset_name, subset_filenames, write):
    '''Output the filenames as a txt file.
    Automatically adds appropriate keras path for the filenames.

    :param dataset_path: The path that the dataset is expected to store under.
                         Defaults to '~/.keras/datasets/'.
    :param dataset_name: The folder that the dataset is expected to store under.
                         Defaults to 'costar_block_stacking_dataset_v0.4'.
    :param dir_path: The path to the directory that the files are in.
    :param dir_name: The name of the directory that the files are in.
    :param category_name: success_only, task_failures_only, error_failures_only, etc.
    :param subset_name: train/val/test, to be added to the output filename.
    :param subset_filenames: The filenames to be written in the txt file.
    :param write: The flag to actually write the output files.
    :return output_path: The path to the output txt file.
    '''
    output_filename = "{0}_{1}_{2}_{3}_files.txt".format(
                        dataset_name, dir_name, category_name, subset_name)
    output_path = os.path.join(dir_path, dir_name, output_filename)
    prefix_path = os.path.join(dataset_path, dataset_name, dir_name)
    print("Output txt file: {}".format(output_path))
    print("Length: {} files".format(len(subset_filenames)))
    print("Example .h5f path in the txt file: %s".format(
          os.path.join(prefix_path, subset_filenames[0])))

    if write:
        with open(output_filename, 'w') as f:
            for filename in subset_filenames:
                file_path = os.path.join(prefix_path, filename)
                f.write(file_path + '\n')
    else:
        print("File not written. Use --write flag to actually output the file.")

    return output_path


def split_dataset(filenames, train_set, val_set, test_set, val_len=None, test_len=None):
    '''Split the input filenames into three sets.
    If val_set and test_set are empty, the sets will be of length val_len and test_len.
    If val_set and test_set have unequal length, match the two lengths.
    Files not in val or test sets are added into training set.

    :param filenames: The filenames to be split into three sets.
    :param train_set: The filenames already in the train set.
    :param val_set: The filenames already in the val set.
    :param test_set: The filenames already in the test set.
    :param val_len: The expected output val set length.
    :param test_len: The expected output test set length.
    :return train_set, val_set, test_set: train/val/test set filenames.
    '''
    if len(test_set) is 0 and test_len is None:
        raise ValueError("split_dataset: test_set is empty and no test_len is specified!")
    if len(val_set) is 0 and val_len is None:
        raise ValueError("split_dataset: val_set is empty and no val_len is specified!")

    # If we reach here without error, either the sets are non-empty, or
    # test_len and val_len are not None
    if test_len is None:
        test_len = len(test_set)
    if val_len is None:
        val_len = len(val_set)

    # No val set and test set is provided, create new val/test sets
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
        # Select filenames not in test or val set
        not_val_or_test_set = [
            filename for filename in filenames if
            filename not in val_set and filename not in test_set]

        # Check if expected length and current lenth for val set are different
        len_diff = val_len - len(val_set)
        if len_diff > 0:
            # Add additional files to val set
            val_set += not_val_or_test_set[0:len_diff]
            not_val_or_test_set = not_val_or_test_set[len_diff:]
            print("Expected val set length: {}, current val set length: {}".format(
                    val_len, len(val_set)))
            print("Added %d files to val set." % len_diff)

            print("Unusual behavior. Do you really want to add files to val set?")
        elif len_diff < 0:
            print("Expected val set length: {}, current val set length: {}".format(
                    val_len, len(val_set)))
            raise RuntimeError(
                "split_dataset: Expected val length is smaller than current length!")

        # Do the same check for test set
        len_diff = test_len - len(test_set)
        if len_diff > 0:
            # Add additional files to test set
            test_set += not_val_or_test_set[0:len_diff]
            not_val_or_test_set = not_val_or_test_set[len_diff:]
            print("Expected test set length: {}, current test set length: {}".format(
                    val_len, len(val_set)))
            print("Added %d files to test set." % len_diff)

            print("Unusual behavior. Do you really want to add files to test set?")
        elif len_diff < 0:
            print("Expected test set length: {}, current test set length: {}".format(
                    val_len, len(val_set)))
            raise RuntimeError(
                "split_dataset: Expected test length is smaller than current length!")

        # Dump the rest of the files into train set
        train_set = not_val_or_test_set

    return train_set, val_set, test_set


# def split_success_only(
#         filenames, path, plush, train_txt, val_txt, test_txt,
#         val_len=None, test_len=None):
#     '''Splits success files into success_only train/val/test txt files.

#     :param filenames: A list of .h5f filenames under the path.
#     :param path: Path to the folder with the .h5f files.
#     :param plush: A bool indicating whether the program is processing plush subset.
#     :param train_txt: Filename to a pre-existing train txt file.
#     :param val_txt: Filename to a pre-existing val txt file.
#     :param test_txt: Filename to a pre-existing test txt file.
#     :param val_len: Expected output val set length.
#     :param test_len: Expected output test set length.
#     :param output_name: Filename prefix to the output train/val/test txt files.
#     '''
#     # Read files that are success
#     filenames = [filename for filename in filenames if '.success.h5f' in filename]
#     print('Selecting ' + str(len(filenames)) + ' success files')

#     # Read filenames for the previous training set
#     if not train_txt:
#         train_set = []
#     else:
#         pre_existing_set_file = path + train_txt
#         if not os.path.isfile(pre_existing_set_file):
#             raise ValueError(
#                 'split_success_only: Pre-existing training file is not a file: ' +
#                 pre_existing_set_file)

#         train_set = get_existing_filenames(pre_existing_set_file)

#     # Read filenames for the previous validation set
#     if not val_txt:
#         val_set = []
#     else:
#         pre_existing_set_file = path + val_txt
#         if not os.path.isfile(pre_existing_set_file):
#             raise ValueError(
#                 'split_success_only: Pre-existing validating file is not a file: ' +
#                 pre_existing_set_file)

#         val_set = get_existing_filenames(pre_existing_set_file)

#     # Read filenames for the previous test set
#     if not test_txt:
#         test_set = []
#     else:
#         pre_existing_set_file = path + test_txt
#         if not os.path.isfile(pre_existing_set_file):
#             raise ValueError(
#                 'split_success_only: Pre-existing testing file is not a file: ' +
#                 pre_existing_set_file)

#         test_set = get_existing_filenames(pre_existing_set_file)

#     # Inform the user that the length of val and test will be matched for output,
#     # when the lengths of val and test are not equal
#     if len(val_set) is not len(test_set):
#         print('Validation set and testing set do not have the same length. '
#               'Output results will be adjusted to same size sets')

#     # Randomize the filenames
#     random.shuffle(filenames)

#     # Split the dataset
#     train_set, val_set, test_set = split_dataset(
#         filenames, train_set, val_set, test_set, val_len, test_len)

#     # Sanity check
#     for i in val_set:
#         if i in train_set:
#             raise RuntimeError("split_success_only: test attempt in train set! %s" % i)
#             # print("split_success_only: val attempt in train set! %s" % i)
#     for i in test_set:
#         if i in train_set:
#             raise RuntimeError("split_success_only: test attempt in train set! %s" % i)
#             # print("split_success_only: test attempt in train set! %s" % i)
#     for i in test_set:
#         if i in val_set:
#             raise RuntimeError("split_success_only: test attempt in val set! %s" % i)
#             # print("split_success_only: test attempt in train set! %s" % i)
#     if (len(train_set) + len(val_set) + len(test_set)) != len(filenames):
#         print("ERROR! lenth of train, val and test = %d, %d, %d"
#               % (len(train_set), len(val_set), len(test_set)))
#         print("Length of all files: %d" % len(filenames))
#         raise RuntimeError("split_success_only: Numbers do not add up. Something is wrong!")
#     print("Split complete. Sanity check passed.")

#     # Write the output files
#     output_file(path, plush, output_name, 'success_only_train', train_set)
#     output_file(path, plush, output_name, 'success_only_val', val_set)
#     output_file(path, plush, output_name, 'success_only_test', test_set)

def read_existing_files(
        dir_path, dataset_name, dir_name, category_name, existing_file_prefix):
    '''Try to open existing train/val/test txt files.
    1. Try [dataset_name]_[dir_name]_[category_name]_[subset_name]_files.txt
       e.g. costar_block_stacking_dataset_v0.4_blocks_only_success_only_train_files.txt
    2. If that does not work, try the prefixes in existing_file_prefix
       e.g. costar_plush_block_stacking_dataset_v0.4_success_only_train_files.txt

    :return train_val_test_filenames: A list of 3 elements representing [train, val, test]
                                      filenames. If no file is found for that subset, the
                                      corresponding element will be None
    '''
    # Read the train/val/test set from dataset name and dir name
    train_val_test_filenames = []
    for subset_name in ['train', 'val', 'test']:
        txt_filename = "{0}_{1}_{2}_{3}_files.txt".format(
                        dataset_name, dir_name, category_name, subset_name)
        txt_file_path = os.path.join(dir_path, txt_filename)
        print("Trying %s..." % txt_file_path)
        if os.path.isfile(txt_file_path):
            print("Existing {} txt file found: {}".format(subset_name, txt_filename))
            train_val_test_filenames.append(get_existing_filenames(txt_file_path))
        else:
            train_val_test_filenames.append(None)

    # If no txt files are found, look in existing_file_prefix
    if any(l is None for l in train_val_test_filenames):
        for prefix in existing_file_prefix:
            train_val_test_filenames = []
            for subset_name in ['train', 'val', 'test']:
                txt_filename = "{0}_{1}_{2}_files.txt".format(
                                prefix, category_name, subset_name)
                txt_file_path = os.path.join(dir_path, txt_filename)
                print("Trying %s..." % txt_file_path)
                if os.path.isfile(txt_file_path):
                    print("Existing {} txt file found: {}".format(
                           subset_name, txt_filename))
                    train_val_test_filenames.append(get_existing_filenames(txt_file_path))
                else:
                    train_val_test_filenames.append(None)

            # Successfully read some pre-existing train/val/test txt files
            if any(l is not None for l in train_val_test_filenames):
                break

    return train_val_test_filenames


def split_sanity_check(train_set, val_set, test_set, len_filenames):
    # Sanity check
    for i in val_set:
        if i in train_set:
            raise RuntimeError("split_sanity_check: test attempt in train set! %s" % i)
    for i in test_set:
        if i in train_set:
            raise RuntimeError("split_sanity_check: test attempt in train set! %s" % i)
    for i in test_set:
        if i in val_set:
            raise RuntimeError("split_sanity_check: test attempt in val set! %s" % i)
    if (len(train_set) + len(val_set) + len(test_set)) != len_filenames:
        print("ERROR! lenth of train, val and test = %d, %d, %d"
              % (len(train_set), len(val_set), len(test_set)))
        print("Length of all files: %d" % len_filenames)
        raise RuntimeError("split_sanity_check: Numbers do not add up!!!")
    print("Sanity check passed.")


def split_all(
        filenames, dataset_name, dir_name, dir_path, success_only, existing_file_prefix,
        val_len=None, test_len=None):
    '''Splits all files into success_only, task_failure_only, error_failure_only, and
    task_and_error_failure subsets.
    1. Open all filenames with h5py to only count the files that contain images
    2. Calculate success:failure:error ratios
    3. Try to open success_only train/val/test txt file in existing_file_prefix for
       length reference. Output train/val/test txt files according to the calculated
       success:failure:error ratio. If no file is found, use val_len and test_len as
       basis for creating success_only train/val/test txt files.

    :param filenames: A list of .h5f filenames under the path.
    :param dataset_name: The name of the dataset.
    :param dir_name: The folder name for that the .h5f files are under.
    :param dir_path: Path to the folder with the .h5f files.
    :param success_only: A bool indicating whether the program should only process
                         success_only files.
    :param existing_file_prefix: Txt file prefixes for the program to look for in
                                 the folder to open as success_only reference.
    :param val_len: Expected output val set length.
    :param test_len: Expected output test set length.
    :return: A list of 4 lists that contain 3 sublists.
             The 4 lists are in the format of [success_only, task_and_error_failure,
             task_failure_only, error_failure_only]
             Each sublist contains [train, val, test] filenames.
    '''
    # Get the success, failure, and error filenames with nonzero frames
    print("Checking h5f files in {}".format(dir_path))
    success_filenames, failure_filenames, error_filenames = count_files_containing_images(
                                                                dir_path, filenames)

    # Calculate the percentage of success, failure and error
    total_file_count = (
        len(success_filenames) + len(failure_filenames) + len(error_filenames))
    success_ratio = len(success_filenames) / total_file_count
    failure_ratio = len(failure_filenames) / total_file_count
    error_ratio = len(error_filenames) / total_file_count
    print("Total: %d files" % total_file_count)
    print("Ratios: {:.2f}% success, {:.2f}% failure(no error), {:.2f}% error".format(
            success_ratio*100, failure_ratio*100, error_ratio*100))

    # Process success_only files
    train_val_test_filenames = read_existing_files(
        dir_path, dataset_name, dir_name, 'success_only', existing_file_prefix)

    # Extract the filenames into subsets
    success_train_set, success_val_set, success_test_set = [
        l if l is not None else [] for l in train_val_test_filenames]
    success_train_len, success_val_len, success_test_len = list(
        map(len, train_val_test_filenames))
    if success_val_len == 0:
        print("No val set found. Default output val length is 64.")
        success_val_len = 64
    if success_test_len == 0:
        print("No test set found. Default output test length is 64.")
        success_test_len = 64

    # Split the dataset
    random.shuffle(success_filenames)
    sucess_train_set, success_val_set, success_test_set = split_dataset(
        success_filenames, success_train_set, success_val_set, success_test_set,
        success_val_len, success_test_len)

    # Sanity check
    print("success_only split complete.")
    split_sanity_check(
        sucess_train_set, success_val_set, success_test_set, len(success_filenames))

    if success_only:
        # If only sucess_only files should be processed, then return the result now
        return [[sucess_train_set, success_val_set, success_test_set], [], [], []]

    # Process .failure.error files
    train_val_test_filenames = read_existing_files(
        dir_path, dataset_name, dir_name, 'error_failure_only', existing_file_prefix)
    err_train_set, err_val_set, err_test_set = [
        l if l is not None else [] for l in train_val_test_filenames]
    # error_train_len, error_val_len, error_test_len = list(
    #     map(len, train_val_test_filenames))

    # Process .failure files
    train_val_test_filenames = read_existing_files(
        dir_path, dataset_name, dir_name, 'task_failure_only', existing_file_prefix)
    fail_train_set, fail_val_set, fail_test_set = [
        l if l is not None else [] for l in train_val_test_filenames]
    # failure_train_len, failure_val_len, failure_test_len = list(
    #     map(len, train_val_test_filenames))

    # Calculate set size for failure and error, based on success_only subset
    multiplier_failure = len(failure_filenames)/len(success_filenames)
    failure_val_len = int(round(success_val_len*multiplier_failure))
    failure_test_len = int(round(success_test_len*multiplier_failure))
    # failure_train_len = len(failure_filenames) - (failure_val_len + failure_test_len)
    multiplier_error = len(error_filenames)/len(success_filenames)
    error_val_len = int(round(success_val_len*multiplier_error))
    error_test_len = int(round(success_test_len*multiplier_error))
    # error_train_len = len(error_filenames) - (error_val_len + error_test_len)

    # Randomize the filenames
    random.shuffle(failure_filenames)
    random.shuffle(error_filenames)

    # Split the dataset for failure and error
    fail_train_set, fail_val_set, fail_test_set = split_dataset(
        failure_filenames, fail_train_set, fail_val_set, fail_test_set,
        failure_val_len, failure_test_len)
    print("task_failure_only split complete.")
    split_sanity_check(
        fail_train_set, fail_val_set, fail_test_set, len(failure_filenames))

    err_train_set, err_val_set, err_test_set = split_dataset(
        error_filenames, err_train_set, err_val_set, err_test_set,
        error_val_len, error_test_len)
    print("error_failure_only split complete.")
    split_sanity_check(
        err_train_set, err_val_set, err_test_set, len(error_filenames))

    # Error is also a type of failure! Combine task failure and error failure subsets.
    task_n_err_train_set = fail_train_set + err_train_set
    task_n_err_val_set = fail_val_set + err_val_set
    task_n_err_test_set = fail_test_set + err_test_set

    return [[sucess_train_set, success_val_set, success_test_set],
            [task_n_err_train_set, task_n_err_val_set, task_n_err_test_set],
            [fail_train_set, fail_val_set, fail_test_set],
            [err_train_set, err_val_set, err_test_set]]


def count_files_containing_images(path, filenames):
    '''Open the files and check frame count. Skip files with 0 frame.

    :param filenames: .h5f filenames in the folder
    :return: Lists of success/failure/error filenames with nonzero frames
    '''
    # TODO: Write total frames into csv file as a new column

    # Open the files to check frame count. Skip files with 0 frame.
    error_filenames = []
    failure_filenames = []
    success_filenames = []
    skip_count = 0
    i = 0
    print("Checking %d files. This can take some time." % len(filenames))
    for filename in filenames:
        i += 1
        if i % 100 == 0:
            # TODO: incorporate tqdm progress bar
            print("{} out of {} files checked".format(i, len(filenames)))
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
                elif 'success' in filename:
                    success_filenames += [filename]
                else:  # BUG: Sanity check for debugging
                    raise Exception(
                        'Somthing is wrong! The file does not contain `error`,'
                        '`failure`, or `success` in the filename: %s' % filename)
        except IOError as ex:
            print('Skipping %s for IO error' % filename)

    print("Counted {:d} success files, {:d} failure files, and {:d} error files.".format(
            len(success_filenames), len(failure_filenames), len(error_filenames)))
    print("Skipped %d files since they have 0 image frame" % skip_count)

    return success_filenames, failure_filenames, error_filenames


def compare_filenames(path, name1, name2):
    '''Check if filenames within two txt files are the same.
    Example use: compare train and val files to make sure the filenames do not overlap.

    :param path: Path containing two txt files to compare.
    :param name1: Filename of a txt file to compare.
    :param name2: Filename of a txt file to compare.
    :return same, diff: Two lists containing the filenames that are the same or different
                        across two txt files.
    '''

    path = os.path.expanduser(path)
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
    path = os.path.expanduser(args['path'])
    if not os.path.isdir(path):
        raise ValueError('Path entered is not a path: ' + path)

    # Get the subfolders under this path
    dir_list = [dir_name for dir_name in os.listdir(path)
                if os.path.isdir(os.path.join(path, dir_name))]

    # Set the random seed for reproducible random lists
    random.seed(args['seed'])

    for dir_name in dir_list:
        dir_path = os.path.join(path, dir_name)

        # Get all h5f files under this folder
        filenames = [filename for filename in os.listdir(dir_path) if '.h5f' in filename]
        if filenames:
            print('Read ' + str(len(filenames)) + ' h5f filenames in the folder')
        else:
            print('Skipping directory %s because it contains no h5f file.' % dir_name)
            continue

        # Split the dataset
        subsets = split_all(
                    filenames, args['dataset_name'], dir_name, dir_path,
                    args['success_only'], args['existing_file_prefix'],
                    args['val_len'], args['test_len'])

        # Output the files
        # Modify here to add more categories
        category_names = [
            'success_only',
            'task_and_error_failure',
            'task_failure_only',
            'error_failure_only']
        subset_names = ['train','val','test']
        for i in range(len(subsets)):
            category_name = category_names[i]
            category_subsets = subsets[i]
            # Output the files
            for j in range(len(category_subsets)):
                subset_name = subset_names[j]
                subset_filenames = category_subsets[j]
                output_file(args['dataset_path'], args['dataset_name'],
                            dir_path, dir_name, category_name,
                            subset_name, subset_filenames)

        # TODO: Implement csv output
        # dataset_splits_csv = 'subset, train_count, val_count, test_count\n'
        # dataset_splits_csv += "success_only, {0}, {1}, {2}\n".format(
        #                         success_train_len, success_val_len, success_test_len)
        # dataset_splits_csv += "task_and_error_failure, {0}, {1}, {2}\n".format(
        #         failure_train_len + error_train_len,
        #         failure_val_len + error_val_len,
        #         failure_test_len + error_test_len)
        # dataset_splits_csv += "task_failure_only, {0}, {1}, {2}\n".format(
        #         failure_train_len, failure_val_len, failure_test_len)
        # dataset_splits_csv += "error_failure_only, {0}, {1}, {2}\n".format(
        #         error_train_len, error_val_len, error_test_len)
        # dataset_splits_csv_filename = 'costar_block_stacking_dataset_split_summary.csv'
        # print(dataset_splits_csv_filename + '\n' + dataset_splits_csv)

        # csv_path = os.path.join(path, dataset_splits_csv_filename)
        # with open(csv_path, 'w+') as file_object:
        #     file_object.write(dataset_splits_csv)


if __name__ == '__main__':
    args = _parse_args()
    main(args)
