import rospy
from threading import Thread

class ServiceCaller(object):
    '''
    Simple helper class built around calling a ROS service
    '''
    def __init__(self, *args, **kwargs):
        self.thread = None
        self.result = None
        self.proxy = None
        self.req = None
        self.running = False

    def _service_call(self):
        if self.proxy is None:
            raise RuntimeError('no proxy specified')
        elif self.req is None:
            raise RuntimeError('no request specified')
        print("RUNNING:", self.proxy, self.req)
        self.result = self.proxy(self.req)

    def __call__(self, proxy, req):
        if self.thread is not None and self.thread.is_alive():
            return False
        else:
            self.proxy = proxy
            self.req = req
            self.thread = Thread(target=self._service_call)
            self.thread.start()
            return True

    def call(self, *args, **kwargs):
        self(*args, **kwargs)

    def update(self):
        if self.thread is None:
            self.running = False
        elif self.thread.is_alive():
            self.running = True
        else:
            self.running = False
        return self.running

