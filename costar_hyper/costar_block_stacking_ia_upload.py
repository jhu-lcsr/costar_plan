""" Internet Archive Dataset Upload Script

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


def timeStamped(fname, fmt='%Y-%m-%d-%H-%M-%S_{fname}'):
    """ Apply a timestamp to the front of a filename description.

    see: http://stackoverflow.com/a/5215012/99379
    """
    return datetime.datetime.now().strftime(fmt).format(fname=fname)


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Upload the dataset to the Internet Archive.')
    parser.add_argument(
        "--path", type=str,
        default=os.path.join(os.path.expanduser("~"),
                             #  '.keras/datasets/costar_block_stacking_dataset_v0.4/'),
                             '/media/ahundt/EA824B88824B5869/costar_block_stacking_dataset_v0.4/'),
        help='Path to dataset folder containing many files. Default is current path.')
    parser.add_argument(
        "--execute", action='store_true', default=False,
        help='Use this flag to actually upload the files to the internet archive')

    return vars(parser.parse_args())


def main(args, root='root'):

    print('If this is your first time running this program:\n'
          ' 1. Make an account at archive.org\n'
          ' 2. On command line, run:\n'
          '    pip install internetarchive\n'
          '    ia configure')
    print('User supplied arguments:\n' + str(args))
    # get the path, and add an extra slash to make sure it is a directory
    # and to avoid a bug in the internet archive upload code.
    path = os.path.expanduser(args['path'] + '/')

    debug = True
    if args['execute']:
        debug = False
        print('WARNING: ATTEMPTING A REAL UPLOAD TO THE INTERNET ARCHIVE. THIS IS NOT A TEST.\n'
              'We are uploading the data to the test_collection, which will only store files for 30 days.\n'
              'When the uplod is complete, email info@archive.org, and they will move your item to a permanent collection.\n'
              'See https://archive.org/about/faqs.php#Collections for details.\n')
    else:
        print('Performing test run.')
    print('debug: ' + str(debug))
    item = internetarchive.get_item('johns_hopkins_costar_dataset', debug=debug)

    md = dict(
        # collection='datasets',
        # You must upload to the test_collection then email them to move your item to another collection.
        # See https://archive.org/about/faqs.php#Collections.
        collection='test_collection',
        title='The Johns Hopkins CoSTAR Robotics Dataset',
        version='v0.4',  # Custom metadata field for the current version
        contributor='Andrew Hundt, Varun Jain, Chris Paxton, Chunting Jiao, '
                    'Chia-Hung Lin, and Gregory D. Hager',
        creator='Andrew Hundt <ATHundt@gmail.com>',
        credits='''
                Andrew Hundt, Varun Jain, Chris Paxton, Chunting Jiao, Chia-Hung Lin,
                and Gregory D. Hager<br>
                The Johns Hopkins University<br>
                <a href="https://cirl.lcsr.jhu.edu/">Computational Interaction and
                Robotics Laboratory</a><br>
                This material is based upon work supported by the National Science
                Foundation under NRI Grant Award No. 1637949.
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
            ''',
        license='https://creativecommons.org/licenses/by/4.0/',
        mediatype='data',  # data is the default media type
        noindex='True')  # Set to true for the item to not be listed

    print('uploading all data from path:\n\n ' + str(path))

    results = item.upload(
        path,
        metadata=md,
        verify=True,  # Verify local MD5 checksum matches remote MD5 checksum
        checksum=True,  # Skip files based on checksum
        verbose=True,  # Print progress to stdout
        retries=100,  # Number of times to retry the given request
        retries_sleep=5,  # Amount of time to sleep between `retries`
        # Prevent an item from being derived to another format after upload
        queue_derive=False,
        # Set to true to print headers to stdout, and exit without uploading
        debug=debug)

    print(results[0].url)
    server_urls = [str(result.url) for result in results]
    local_urls = [str(result.path_url) for result in results]

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

if __name__ == '__main__':
    args = _parse_args()
    main(args)
