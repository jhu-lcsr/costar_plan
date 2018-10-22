from block_stacking_reader import CostarBlockStackingSequence
import h5py
import os
import tensorflow as tf
import numpy as np
import glob
from keras.models import model_from_json
from grasp_utilities import load_hyperparams_json
from cornell_grasp_train import get_compiled_model
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class CostarHyperTreeInference():

    def __init__(self, filenames, hyperparams_json, load_weights, problem_name):
        self.filenames = filenames
        self.hyperparams_json = hyperparams_json
        self.problem_name = problem_name
        self.load_weights = load_weights
        self.gripper_action_goal_idx = []
        self.inference_mode_gen(self.filenames)

    def inference_mode_gen(self, file_names):
        """ Generate data for all time steps in a single example.
        """
        self.file_list_updated = []
        self.file_len_list = []
        # print(len(file_names))
        file_mode = "w"
        file_len = 0
        print('len ', len(file_names))
        for f_name in file_names:
            with h5py.File(f_name, 'r') as data:
                file_len = len(data['gripper_action_goal_idx']) - 1
                self.file_len_list.append(file_len)
                self.gripper_action_goal_idx.append(list(data['gripper_action_goal_idx']))

        for i in range(len(file_names)):
            for j in range(self.file_len_list[i]):
                self.file_list_updated.append(file_names[i])
        # return file_list_updated, file_len_list

    def block_stacking_generator(self, sequence):
        epoch_size = 1
        step = 0
        while True:
            # if step > epoch_size:
            #     step = 0
            #     sequence.on_epoch_end()
            batch = sequence.__getitem__(step)
            step += 1
            yield batch
    def evaluateModel(self, generator):
        filenames_updated, file_len_list = self.file_list_updated, self.file_len_list
        hyperparams = load_hyperparams_json(self.hyperparams_json)
        hyperparams.pop('checkpoint', None)
        model = get_compiled_model(**hyperparams, problem_name=self.problem_name, load_weights=self.load_weights)
        bsg = self.block_stacking_generator(generator)
        with open("inference_results_per_frame.csv", 'w') as fp:
            cw = csv.writer(fp, delimiter=',', lineterminator='\n')
            cw.writerow(['example', 'frame_no'] + model.metrics_names)
            # fp.write("\n")
        frame_counter = 0
        file_counter = 0
        frame_len = file_len_list[file_counter]
        for i in range(len(generator)):
            data = next(bsg)
            if filenames_updated[i] != filenames[file_counter]:
                file_counter += 1
                frame_len = file_len_list[file_counter]
                frame_counter = 0

            # print("len of X---", len(data[0]))
            frame_counter += 1 % frame_len
            score = model.evaluate(data[0], data[1])
            with open("inference_results_per_frame.csv", 'a') as fp:
                cw = csv.writer(fp, delimiter=',', lineterminator='\n')
                score = [file_counter] + [frame_counter] + score
                cw.writerow(score)
    def plotErrorFrameDist(self, score_file, metric_2):
        with open(score_file, 'r') as fp:
            reader = csv.reader(fp)
            headers = next(reader, None)
            scores = list(reader)
        # metric_1_index = headers.index(metric_1)
        metric_2_index = headers.index(metric_2)
        frames = []
        loss = []
        for row in scores:
            frames.append(row[1])
            loss.append(row[metric_2_index])

        print(headers)
        frames = list(map(int, frames))
        loss = list(map(float, loss))
        figure1 = plt.figure(1, figsize=(20, 10))
        plt.xticks(np.arange(min(frames), max(frames)+1, 10))
        plt.yticks(np.arange(min(loss), max(loss)+1, 0.1))
        indexes = np.where(np.array(frames) == 1)[0]
        # print(indexes)
        ax = plt.axes()
        n_lines = len(indexes)
        ax.set_color_cycle([plt.cm.cool(i) for i in np.linspace(0, 1, n_lines)])
        count = 0
        for i in indexes[1:]:
            goals = self.gripper_action_goal_idx[count]
            count += 1
            # print(len(goals))
            # print(len(frames[indexes[count-1]:i]))
            plt.scatter(np.array(goals[1:] - np.array(frames[indexes[count-1]:i])), loss[indexes[count-1]:i])
        goals = self.gripper_action_goal_idx[-1]
        # print(goals)
        plt.scatter(np.array(goals[1:]) - frames[indexes[-1]:-8], loss[indexes[-1]:-8])
        # print(frames[indexes[-1]:-8]-np.array(goals[1:]))
        # plt.plot(frames[:225], loss[:225])
        # plt.plot(frames[225:], loss[225:])
        plt.xlabel('Distance to goal')
        plt.ylabel(metric_2)
        plt.savefig("plot1.png")
        plt.show()
        # plt.show()
        figure2 = plt.figure(2, figsize=(20, 10))
        frame_range = range(1, len(frames)+1)
        print(len(frame_range))
        plt.xticks(np.arange(min(loss), max(loss)+1, 0.1))
        figure2.axes[0].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=len(frames)))
        plt.plot(np.sort(loss), frame_range)
        # print(figure2.axes)
        plt.xlabel(metric_2)
        plt.ylabel('Examples')
        plt.savefig('plot2.png')
        plt.show()


if __name__ == "__main__":

    filenames = glob.glob(os.path.expanduser(r'C:\Users\Varun\JHU\LAB\Projects\costar_task_planning_stacking_dataset_v0.1/*success.h5f'))
    output_shape = (224, 224, 3)
    load_weights = "2018-09-04-20-17-25_train_v0.3_msle-vgg_semantic_rotation_regression_model--dataset_costar_block_stacking-grasp_goal_aaxyz_nsc_5-epoch-412-val_loss-0.002-val_angle_error-0.279.h5"
    # filenames_updated, file_len_list = inference_mode_gen(filenames[:2])
    print(len(filenames))
    training_generator = CostarBlockStackingSequence(
        filenames[:4], batch_size=1, verbose=1,
        output_shape=output_shape,
        label_features_to_extract='grasp_goal_aaxyz_nsc_5',
        data_features_to_extract=['current_xyz_aaxyz_nsc_8'],
        blend_previous_goal_images=False, inference_mode=True, pose_name="pose")
    # bsg = block_stacking_generator(training_generator)
    # print('bsg len', len(training_generator))
    file_mode = 'w'
    hyperparams = "2018-09-04-20-17-25_train_v0.3_msle-vgg_semantic_rotation_regression_model--dataset_costar_block_stacking-grasp_goal_aaxyz_nsc_5_hyperparams.json"
    problem_name = 'semantic_rotation_regression'
    hypertree_inference = CostarHyperTreeInference(filenames=filenames, hyperparams_json=hyperparams, load_weights=load_weights, problem_name=problem_name)
    hypertree_inference.evaluateModel(training_generator)
    hypertree_inference.plotErrorFrameDist('inference_results_per_frame.csv', 'angle_error')
