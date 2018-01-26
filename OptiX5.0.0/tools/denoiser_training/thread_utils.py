#----------------------------------------------------------------------------
# thread_utils.py: Multithreading utilities.
#----------------------------------------------------------------------------

import threading, Queue, sys, traceback

#----------------------------------------------------------------------------

class ExceptionInfo(object):
    def __init__(self):
        self.type, self.value = sys.exc_info()[:2]
        self.traceback = traceback.format_exc()

#----------------------------------------------------------------------------

class WorkerThread(threading.Thread):
    def __init__(self, task_queue):
        threading.Thread.__init__(self)
        self.task_queue = task_queue

    def run(self):
        while True:
            func, args, result_queue = self.task_queue.get()
            if func is None:
                break
            try:
                result = func(*args)
            except:
                result = ExceptionInfo()
            result_queue.put((result, args))

#----------------------------------------------------------------------------

class ThreadPool(object):
    def __init__(self, num_threads):
        assert num_threads >= 1
        self.task_queue = Queue.Queue()
        self.result_queues = {}
        self.num_threads = num_threads
        for idx in xrange(self.num_threads):
            WorkerThread(self.task_queue).start()

    def add_task(self, func, args = ()):
        assert hasattr(func, '__call__') # must be a function
        if func not in self.result_queues:
            self.result_queues[func] = Queue.Queue()
        self.task_queue.put((func, args, self.result_queues[func]))

    def get_result(self, func, verbose_exceptions = True): # returns (result, args)
        result, args = self.result_queues[func].get()
        if isinstance(result, ExceptionInfo):
            if verbose_exceptions:
                print '\n\nWorker thread caught an exception:\n' + result.traceback + '\n',
            raise result.type, result.value
        return result, args

    def finish(self):
        for idx in xrange(self.num_threads):
            self.task_queue.put((None, (), None))

    def __enter__(self): # for 'with' statement
        return self

    def __exit__(self, *excinfo):
        self.finish()

#----------------------------------------------------------------------------

def run_iterator_concurrently(iterator, thread_pool):

    def task_func():
        try:
            result = iterator.next()
        except StopIteration as exc:
            result = exc
        return result

    thread_pool.add_task(task_func)
    while True:
        result, args = thread_pool.get_result(task_func)
        if isinstance(result, StopIteration):
            break
        thread_pool.add_task(task_func)
        yield result

#----------------------------------------------------------------------------
