import tensorflow as tf
import os.path
from grasp_dataset import GraspDataset

""" Counting success, failure, success ratio to total.
    Write to ~/.keras/dataset/num/grasping'.
"""
if __name__ == '__main__':
    with tf.Session() as sess:
        gd = GraspDataset()
        default_save_path = os.path.join(
            os.path.expanduser('~'), '.keras', 'datasets', gd.dataset, 'grasping')
        os.makedirs(default_save_path)
        filename = 'grasp_dataset_success_statistics.txt'
        complete_path = os.path.join(default_save_path, filename)
        success_num, fail_num, success_ratio = gd.count_success_failure_number(sess)
        file_object = open(complete_path, 'w')
        text_lines = ['Statistics for grasp_dataset_', gd.dataset+'\n',
                      'total attempts: ', str(success_num+fail_num)+'\n',
                      'number of success: ', str(success_num)+'\n',
                      'number of failure: ', str(fail_num)+'\n',
                      'success ratio to total: ', str(success_ratio)+'\n']
        file_object.writelines(text_lines)
        file_object.close()
        print('number of attempts', success_num+fail_num)
        print('number of success', success_num)
        print('number of failure', fail_num)
        print('success ratio to total', success_ratio)
