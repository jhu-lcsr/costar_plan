""" Internet Archive Dataset Single File Upload Script

The Internet Archive Python Library and Command Line Tool is at:
    https://github.com/jjjake/internetarchive

However, we recommend installing the internet archive library version at:
    https://github.com/RexxarCHL/internetarchive

Until https://github.com/jjjake/internetarchive/pull/274 has been merged.
"""
import internetarchive
import argparse
import os
import datetime
import numpy as np
from tqdm import tqdm  # tqdm required for pb.write()
from glob import glob


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Upload the dataset to the Internet Archive.')
    parser.add_argument(
        "--path", type=str,
        # default=os.path.join(os.path.expanduser("~"),
        #                      '.keras/datasets/costar_block_stacking_dataset_v0.4/'),
        default='/media/ahundt/EA824B88824B5869/costar_block_stacking_dataset_v0.4/',
        help='Path to dataset folder containing many files. '
             'Default is .keras/datasets/costar_block_stacking_dataset_v0.4/')
    parser.add_argument(
        '--files_hash_csv', type=str,
        default='costar_upload_files_hash.csv',
        help='The filename of the CSV that contains uploaded files and their hashes.')
    parser.add_argument(
        '--exclude_files_txt', type=str,
        default='costar_upload_exclude_files.txt',
        help='''The txt file that contains filenames to be excluded in the upload process.
             The file can contain the relative path to the file to be excluded, or a glob
             pattern. Each line must be separated with a end-of-line character(\\n)''')
    parser.add_argument(
        "--execute", action='store_true', default=False,
        help='Use this flag to actually upload the files to the internet archive')
    parser.add_argument(
        "--include_ext", type=str, nargs='+',
        default=['.txt', '.h5f', '.csv', '.yaml'],
        help='File extensions for selecting files to upload. '
             'Default is .txt, .h5f, .csv, and .yaml')

    return vars(parser.parse_args())


def timeStamped(fname, fmt='%Y-%m-%d-%H-%M-%S_{fname}'):
    """ Apply a timestamp to the front of a filename description.

    see: http://stackoverflow.com/a/5215012/99379
    """
    return datetime.datetime.now().strftime(fmt).format(fname=fname)


def save_file_hash_as_csv(csv_path, file_hash_table):
    '''Save the filename hash table as a CSV file
    '''
    with open(csv_path, 'wb') as csv_file:
        np.savetxt(csv_file, file_hash_table, fmt='%s', delimiter=', ',
                   header='filename, md5_hash')


def main(args, root='root'):

    print('If this is your first time running this program:\n'
          ' 1. Make an account at archive.org\n'
          ' 2. On command line, run:\n'
          '    pip install internetarchive\n'
          '    ia configure')
    print('User supplied arguments:\n' + str(args))

    # Ensure that the path argument is an directory, to avoid a bug in the internetarchive
    # library in processing directories without trailing slash.
    path = os.path.expanduser(args['path'])
    if path[-1] != '/' or not os.path.isdir(path):
        print('ERROR: We only accept directories in this script so'
              ' there must be a trailing slash /. Try:\n'
              '   python costar_block_stacking_ia_upload.py --path ' + str(path) + '/'
              '\nExiting.')
        return

    debug = True
    if args['execute']:
        debug = False
        print('\n\nWARNING: ATTEMPTING A REAL UPLOAD TO THE INTERNET ARCHIVE. THIS IS NOT A TEST.\n\n'
              'We are uploading the data to the test_collection, which will only store files for 30 days.\n'
              'When the uplod is complete, email info@archive.org, and they will move your item to a permanent collection.\n'
              'See https://archive.org/about/faqs.php#Collections for details.\n')
    else:
        print('Performing test run.')
    print('debug: ' + str(debug))

    # Get the list of excluded files
    txt_path = os.path.join(path, args['exclude_files_txt'])
    if os.path.isfile(txt_path):
        glob_list = np.genfromtxt(txt_path, dtype='str', delimiter='\n')
        print('Imported {} glob rules from {}'.format(len(glob_list), txt_path))
    else:
        with open(txt_path, 'w') as f:
            print('Created exclude file txt at {}'.format(txt_path))
        glob_list = []
    excluded_list = []
    for s in glob_list:
        excluded_list += glob(s)
    print('Selected {} files to be excluded'.format(len(excluded_list)))

    # Get the path to all h5f, txt, and csv files in the directory and subdirectories
    filenames = []
    include_ext = args['include_ext']
    print('Selecting files with extensions: \n{}'.format(str(include_ext)))
    for root, _, files in os.walk(path):
        for filename in files:
            # Skip the hash CSV and excluded files
            if (args['files_hash_csv'] in filename or
               filename in excluded_list):
                continue
            # Only select proper extensions
            if any([ext in filename for ext in include_ext]):
                rel_dir = os.path.relpath(root, path)
                if rel_dir == '.':
                    rel_dir = ''
                filenames.append(os.path.join(rel_dir, filename))
    if len(filenames) == 0:
        raise RuntimeError('No matching files found! '
                           'Are you sure the path is correct? {}'.format(path))
    print('Counted {} matching files in \n{}'.format(len(filenames), path))

    # Read in the current uploaded files from a CSV file
    csv_path = os.path.join(path, args['files_hash_csv'])
    if os.path.isfile(csv_path):
        print("Loading existing filename hash CSV file: \n{}".format(csv_path))
        file_hash_table = np.genfromtxt(csv_path, dtype='str', delimiter=', ')
    else:
        print("Creating a new filename hash CSV file.: \n{}".format(csv_path))
        file_hash_table = np.column_stack(
            [filenames, ['not_uploaded_yet'] * len(filenames)])
        save_file_hash_as_csv(csv_path, file_hash_table)

    if file_hash_table.shape[0] < len(filenames):
        diff = len(filenames) - file_hash_table.shape[0]
        print('Adding {} new files to the CSV file...'.format(diff))
        for filename in filenames:
            if filename in file_hash_table:
                continue
            print('Added {}'.format(filename))
            file_hash_table = np.append(file_hash_table,
                                        [[filename, 'not_uploaded_yet']],
                                        axis=0)
        # Save the newly added contents
        save_file_hash_as_csv(csv_path, file_hash_table)
    elif file_hash_table.shape[0] > len(filenames):
        # TODO(rexxarchl): Handle this case
        raise RuntimeError('File count({}) is greater than actual file count({})!'.format(
            file_hash_table.shape[0], len(filenames)))

    # Get the item from the internetarchive
    # item = internetarchive.get_item('johns_hopkins_costar_dataset', debug=debug)
    item = internetarchive.get_item('deleteme_10-23-2018', debug=debug)

    # Define the metadata
    md = dict(
        # collection='datasets',
        # You must upload to the test_collection then email them to move your item to
        # another collection.
        # See https://archive.org/about/faqs.php#Collections.
        collection='test_collection',
        title='The Johns Hopkins CoSTAR Robotics Dataset',
        version='v0.4',  # Custom metadata field for the current version
        contributor='Andrew Hundt, Varun Jain, Chris Paxton, Chunting Jiao, '
                    'Chia-Hung Lin, and Gregory D. Hager',
        creator='Andrew Hundt <ATHundt@gmail.com>',
        credits='''
                Andrew Hundt, Varun Jain, Chris Paxton, Chunting Jiao, Chia-Hung Lin, and Gregory D. Hager<br>
                The Johns Hopkins University<br>
                <a href="https://cirl.lcsr.jhu.edu/">Computational Interaction and Robotics Laboratory</a><br>
                This material is based upon work supported by the National Science Foundation under NRI Grant Award No. 1637949.
                ''',
        date='2018-10-19',
        description='''
            Stack blocks like a champion! The CoSTAR Block Stacking Dataset includes a
            real robot trying to stack colored children's blocks more than 10,000 times
            in a scene with challenging lighting and a movable bin obstacle which must
            be avoided. This dataset is especially well suited to the benchmarking and
            comparison of deep learning algorithms.<br>
            Visit the <a href='https://sites.google.com/site/costardataset'>CoSTAR Dataset Website</a> for more info.<br>
            <b>If you use the dataset, please cite our paper introducing it:</b>
            <a href='https://sites.google.com/view/hypertree-renas'>Training Frankenstein's Creature to Stack: HyperTree Architecture Search</a>

            Andrew Hundt, Varun Jain, Chris Paxton, Chunting Jiao, Chia-Hung Lin, and Gregory D. Hager<br>
            The Johns Hopkins University<br>
            <a href="https://cirl.lcsr.jhu.edu/">Computational Interaction and Robotics Laboratory</a><br>
            This material is based upon work supported by the National Science Foundation under NRI Grant Award No. 1637949.
            ''',
        license='https://creativecommons.org/licenses/by/4.0/',
        mediatype='data',  # data is the default media type
        noindex='True')  # Set to true for the item to not be listed

    print('Uploading all data in the following directory:\n ' + str(path))
    success_count, failed_count, skip_count = 0, 0, 0
    hash_csv_idx = -1
    results_url = []
    results_path_url = []
    pb = tqdm(range(len(filenames)))
    for i in pb:
        file_path, md5_hash = file_hash_table[i]
        pb.write('Uploading {}'.format(file_path))
        if args['files_hash_csv'] in file_path:
            # skip_count += 1
            hash_csv_idx = i
            continue  # Skip the file hash until the end
        if md5_hash != 'not_uploaded_yet':
            skip_count += 1
            pb.write('Skipping {} because it has been uploaded'.format(file_path))
            continue  # Skip uploaded files

        # Upload the file
        resp = item.upload_file(
            os.path.join(path, file_path),
            key=file_path,
            metadata=md,
            verify=True,
            checksum=True,
            retries=10,
            retries_sleep=30,
            queue_derive=False,
            debug=debug)

        # Check if the file is successfully uploaded
        # A successful upload should have status_code = 200
        if debug:
            pb.write('[DEBUG] item key = {}'.format(file_path))
            results_url.append(resp.url)
            results_path_url.append(resp.path_url)
            with open(os.path.join(path, file_path), 'rb') as f:
                md5_hash = internetarchive.utils.get_md5(f)
            success_count += 1
        elif resp.status_code is None:
            # NOTE: Response object is empty. This file is already on the server.
            # See definition of upload_file in internetarchive/item.py
            pb.write('{} is already on the server.'.format(
                file_path))
            # File already on server. Record the hash
            with open(os.path.join(path, file_path), 'rb') as f:
                md5_hash = internetarchive.utils.get_md5(f)
            skip_count += 1
        elif resp.status_code != 200:
            pb.write('Upload failed for {}, status code = {}'.format(
                file_path, resp.status_code))
            failed_count += 1
        else:
            results_url.append(resp.request.url)
            results_path_url.append(resp.request.path_url)
            # File successfully sent to server. Record the hash
            with open(os.path.join(path, file_path), 'rb') as f:
                md5_hash = internetarchive.utils.get_md5(f)
            success_count += 1
        file_hash_table[i] = np.array([file_path, md5_hash])

        if (success_count + 1) % 10 == 0:
            pb.write(timeStamped('[%d] Check point, saving csv' % i))
            save_file_hash_as_csv(csv_path, file_hash_table)

    print('Uploaded {} files. Skipped {} files. {} files failed to upload.'.format(
        success_count, skip_count, failed_count))
    print('Total file count {}, expected file count {}'.format(
        success_count+skip_count+failed_count, len(filenames)))

    print('Saving file hash CSV file for uploading.')
    with open(csv_path, 'rb') as f:
            md5_hash = internetarchive.utils.get_md5(f)
    file_hash_table[hash_csv_idx][1] = md5_hash
    save_file_hash_as_csv(csv_path, file_hash_table)
    # Upload the file
    resp = item.upload_file(
        csv_path,
        key=args['files_hash_csv'],
        metadata=md,
        verify=True,
        checksum=True,
        retries=10,
        retries_sleep=30,
        queue_derive=False,
        debug=debug)
    if debug:
        print('[DEBUG] item key = {}'.format(file_path))
    elif resp.status_code != 200:
        print('Upload failed for {}, status code = {}'.format(
            csv_path, resp.status_code))
        failed_count += 1
    else:
        results_url.append(resp.request.url)
        results_path_url.append(resp.request.path_url)

    print('Upload finished, printing the results:')
    server_urls = [url for url in results_url]
    local_urls = [path_url for path_url in results_path_url]

    if debug:
        debug_str = '_debug'
    else:
        debug_str = ''
    prefix = timeStamped('internet_archive_uploaded' + debug_str)
    server_txt = prefix + '_server_urls.txt'
    local_txt = prefix + '_local_path_urls.txt'
    with open(server_txt, mode='w') as set_file:
        set_file.write('\n'.join(server_urls))
    with open(local_txt, mode='w') as set_file:
        set_file.write('\n'.join(local_urls))
    print('-' * 80)
    print('local_urls:' + str(local_urls))
    print('-' * 80)
    print('server_urls:' + str(server_urls))
    print('-' * 80)
    print('internet archive upload complete! file lists are in: \n' +
          str(server_txt) + '\n' + str(local_txt))
    if failed_count != 0:
        print('{} files failed to upload! Re-run the script to try again'.format(
            failed_count))


if __name__ == '__main__':
    args = _parse_args()
    main(args)
