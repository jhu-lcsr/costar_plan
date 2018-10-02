import internetarchive
import argparse
import os

def _parse_args():
    parser = argparse.ArgumentParser(description=\
        'Uploads the folder specified in `path` argument to the Internet Archive.')
    parser.add_argument("--path", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ''), help='Path to dataset folder containing many files. Default is current path.')
    parser.add_argument("--execute", action='store_true', default=False, help='Use this flag to actually upload the files to the internet archive')
    
    return vars(parser.parse_args())


def main(args, root = 'root'):
    item = internetarchive.get_item('costar_block_stacking_dataset')
    
    md = dict(collection='test_collection', title='The CoSTAR Block Stacking Dataset', mediatype='data', noindex='True')
    
    print args

    r = item.upload(
            args['path'], 
            metadata = md,
            verify = True, # Verify local MD5 checksum matches remote MD5 checksum
            checksum = True, # Skip files based on checksum
            verbose = True, # Print progress to stdout
            retries = 100, # Number of times to retry the given request
            retries_sleep = 5, # Amount of time to sleep between `retries`
            queue_derive = False, # Prevent an item from being derived to another format after upload
            # debug = args['execute']) # Set to true to print headers to stdout, and exit without uploading
            debug = True)
            
    print r

if __name__ == '__main__':
    args = _parse_args()
    main(args)
