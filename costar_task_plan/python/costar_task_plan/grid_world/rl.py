
from world import *


def RunQlearningIter(x_var,  # tensor flow variable for input/state
                     y_var,  # tensor flow variable for output/action
                     r_var,  # tensor flow variable for reward/target
                     x_dim,
                     # numer of input variables: dimensionality of s in Q(s,a)
                     y_dim,
                     # number of output variables: dimensionality of a in
                     # Q(s,a)
                     y_out,  # output actions
                     train_step,  # training step for this function
                     get_world,  # function to create a new world to simulate
                     evaluate_world,
                     # function called to run test and get rewards
                     get_features,
                     # function pointer to retrieve feature vectors
                     experience,  # experience memory
                     accuracy=None,
                     # metric for performance accuracy on state/actions
                     valid_x_data=None,
                     valid_y_data=None,
                     r_out=None,
                     num_simulations=100,
                     # number of worlds to create and test
                     world_iter=100,  # iterations to simulate forward in time
                     retrain_iter=1000,  # how much training do we do
                     retrain_sample_size=1000,  # how many samples do we draw
                     index=0
                     ):
    '''
    Create a random world and see how our actor does!
    '''
    useModelActor = True
    for w in xrange(num_simulations):

        (world, actor) = get_world(x_var, x_dim, y_out)

        (score, res, memory) = evaluate_world(
            world, actor, x_dim, y_dim, get_features, world_iter)
        print '[%d] Test result = %f for reason "%s"' % (w, score, res)

        experience.addInput(memory)

    # print "Total samples = %d"%experience._length

    '''
    RETRAIN
    '''

    # train using all data for 1000 steps
    for i in xrange(retrain_iter):
        # idx = random.sample(range(xdata.shape[0]),2000)
        (xin, yin, rin) = experience.sample(retrain_sample_size)
        train_step.run(feed_dict={x_var: xin, y_var: yin, r_var: rin})
        if accuracy is not None and (index + i) % 100 == 0:
            print "-- iter %d accuracy = %f, train accuracy = %f" % (i,
                                                                     accuracy.eval(
                                                                     feed_dict={
                                                                         x_var: valid_x_data, y_var: valid_y_data}),
                                                                     accuracy.eval(
                                                                     feed_dict={x_var: experience._prev_x, y_var: experience._y})
                                                                     )
    # if r_out is not None:
    #    print rin
    #    print r_out.eval(feed_dict={x_var: xin, y_var: yin})
