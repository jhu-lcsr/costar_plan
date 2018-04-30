import rospy
from threading import Thread
import traceback

class ServiceCaller(object):
    '''
    Simple helper class built around calling a ROS service.
    Typically this will handle executing an action and receiving
    the reply, such as a SmartGrasp or SmartRelease request.
    '''
    def __init__(self, *args, **kwargs):
        self.thread = None
        self.result = None
        self.proxy = None
        self.req = None
        self.running = False
        self.ok = True

    def _service_call(self):
        if self.proxy is None:
            raise RuntimeError('service_caller.py ServiceCaller no proxy specified')
        elif self.req is None:
            raise RuntimeError('service_caller.py ServiceCaller no request specified')
        self.result = self.proxy(self.req)
        self.ok = self.result is not None and "success" in self.result.ack.lower()

    def __call__(self, proxy, req):
        """ Run the service call, i.e. execute the chosen action
        """
        if req is None:
            rospy.logerr('service_caller.py ServiceCaller received a request'
                         ' with the invalid python value None:\n'
                         ''.join(traceback.format_stack()))
        if self.thread is not None and self.thread.is_alive():
            rospy.logwarn("already running: " + str(self.proxy) + ", " + str(type(req)))
            return False
        else:
            self.proxy = proxy
            self.req = req
            self.ok = True
            self.thread = Thread(target=self._service_call)
            self.thread.start()
            return True

    def call(self, *args, **kwargs):
        self(*args, **kwargs)

    def reset(self):
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        self.running = False
        self.ok = True

    def update(self):
        if self.thread is None:
            self.running = False
        elif self.thread.is_alive():
            self.running = True
        else:
            self.thread = None
            self.running = False
        return self.running

