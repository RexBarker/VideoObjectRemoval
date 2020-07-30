from threading import Thread
from time import time, sleep
import paramiko

class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        #print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return

def boring(waittime):
    stdin, stdout, stderr = client.exec_command(f"sleep {waittime}; echo I did something")
    exit_status = stdout.channel.recv_exit_status()
    return [l.strip() for l in stdout]


client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('inpaint', username='appuser', password='appuser') 

#stdin, stdout, stderr = client.exec_command('ls -l')
start = time()
t1 = ThreadWithReturnValue(target=boring,args=(10,)) 
t1.start()
#t1.join()

while t1.is_alive():
    print("apparently not done")
    sleep(1)

res = t1.join()
finish = time()
print("That took ",finish - start, " seconds")

print("\n".join(res))


client.close()