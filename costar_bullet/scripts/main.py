'''
By Chris Paxton
(c) 2017 The Johns Hopkins University
See License for Details

This tool is designed to allow you to bring up task planning and reinforcement
learning experiments using the CTP library and Bullet3 physics. For
instructions see the readme.
'''

from costar_task_plan.simulation import ParseBulletArgs
from costar_task_plan.gym import BulletSimulationEnv
from costar_task_plan.agent import GetAgents, MakeAgent
from costar_task_plan.models import MakeModel

import time

def main(args):
    if 'cpu' in args and args['cpu']:
        import tensorflow as tf
        import keras.backend as K

        with tf.device('/cpu:0'):
            config = tf.ConfigProto(
                device_count={'GPU': 0}
            )
            sess = tf.Session(config=config)
            K.set_session(sess)

    env = BulletSimulationEnv(**args)
    if 'agent' in args and args['agent'] is not None:
        agent = MakeAgent(env, args['agent'], **args)
        agent.fit(num_iter=args['iter'])
    if 'model' in args and args['model'] is not None:
        model = MakeModel(taskdef=env.task, **args)
        if 'load_model' in args and args['load_model']:
            model.load(env.world)
        try:
            model.train(**agent.data)
        except KeyboardInterrupt, e:
            pass
        model.save()
        if args['debug_model']:
            model.plot(env)
            try:
                while True:
                    time.sleep(0.1)
            except Exception, e:
                pass
    else:
        pass

if __name__ == '__main__':
    '''
    This simple tool should parse arguments and pass them to a simulation client.
    The client actually manages everything, including the ROS core.
    '''

    args = ParseBulletArgs()
    if args['profile']:
        import cProfile
        cProfile.run('main(args)')
    else:
        main(args)
