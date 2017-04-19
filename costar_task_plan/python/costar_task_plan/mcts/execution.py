
# This is a plan for executing a predefined sequence of options.
class ExecutionPlan(object):

    def __init__(self, plan, verbose=True, actor_id=0):
        self.nodes = []

        print "Creating execution plan from tree search results:"
        for i, node in enumerate(plan):
            print "node =", node.tag, 
            
            # check node
            if node.action is None and i == 0:
                print " -- skipping root"
                continue
            elif node.action is None:
                raise RuntimeError('execution plan must contain a whole sequence of actions')

            # check policy
            if node.action.policy is None:
                raise RuntimeError('execution plan must contain a whole sequence of policies')

            # check condition
            if node.action.condition is None:
                raise RuntimeError('execution plan must contain a whole sequence of conditions')

            print "action =", node.action.tag
            self.nodes.append(node)

        self.reset(actor_id)

    # set counters to zero
    def reset(self, actor_id=0):
        self.idx = 0
        self.world = None
        self.actor_id = actor_id

    
    def step(self, world):
        actor = world.actors[self.actor_id]
        state = actor.state
        cmd = None
        done = world.done

        # Check to see if we can even be executing at this point. If not, then
        # we should just quit and return.
        if done:
          return True

        while cmd is None and self.idx < len(self.nodes):
            # if we are advancing along a trajectory get the next action...
            condition = self.nodes[self.idx].action.condition
            if condition(world, state, actor, actor.last_state):
                policy = self.nodes[self.idx].action.policy
                cmd = policy.evaluate(world, state, actor)
            else:
              cmd = None
              self.idx += 1

        if cmd is None:
            cmd = world.zeroAction()
            done = True

        # Send the command
        import rospy
        rospy.logwarn(self.nodes[self.idx].tag + ", command is: " + str(cmd.dq))
        rospy.loginfo("world is terminal?" + str(done) + ", " + str(world.done))
        
        # If we did not perform any other action, just update and tick the
        # world. Otherwise cmd will hold the value of the next policy.
        world.tick(cmd)
        return done
