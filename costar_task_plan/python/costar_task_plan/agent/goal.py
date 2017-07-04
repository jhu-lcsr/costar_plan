'''
By Chris Paxton
Copyright (c) 2017, The Johns Hopkins University
All rights reserved.

This license is for non-commercial use only, and applies to the following
people associated with schools, universities, and non-profit research institutions

Redistribution and use in source and binary forms by the aforementioned
people and institutions, with or without modification, are permitted
provided that the following conditions are met:

* Usage is non-commercial.

* Redistribution should be to the listed entities only.

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''


from abstract import AbstractAgent


class RandomGoalAgent(AbstractAgent):
    '''
    Reads goal information from the task and world. Will sample goals based
    on some guiding information provided by the goal and the task model.

    Unlike the others, this relies on having a working implementation of the
    "simulation.robot.RobotInterface" abstract class associated with the task.
    '''

    name = "random_goal"

    def __init__(self, max_iter=10000, *args, **kwargs):
        super(RandomGoalAgent, self).__init__(*args, **kwargs)
        self.max_iter = max_iter
        self.task = None
        self.task_def = None

    def fit(self, env):
        '''
        Set the task and task model. We will sample from the task model at
        each step to move to our goals, with a random walk. We'll use any hints
        provided in the task model to help make sure we get somewhat useful
        information.
        '''
        self.task_def = env.client.task
        self.task = env.client.task.task
        for i in xrange(10000):
            cmd = env.action_space.sample()
            env.step(cmd)

        return None
        
