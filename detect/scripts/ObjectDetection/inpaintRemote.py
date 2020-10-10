# Utilities to run DeepFlow Inpaint from remote container
from time import time
from paramiko import SSHClient, AutoAddPolicy
from threading import Thread

# primarily utilize parent methods, where possible
class InpaintRemote(SSHClient):
    def __init__(self, *args, **kwargs):
        super(InpaintRemote,self).__init__(*args, **kwargs)
        self.isConnected = False
        self.set_missing_host_key_policy(AutoAddPolicy())
        self.c = { 
                   'pythonPath': "/usr/bin/python3",
                   'workingDir': "/home/appuser/Deep-Flow",
                   'scriptPath': "/home/appuser/Deep-Flow/tools/video_inpaint.py",
                   'pretrainedModel': "/home/appuser/Deep-Flow/pretrained_models/FlowNet2_checkpoint.pth.tar",
                   'optionsString' : "--FlowNet2 --DFC --ResNet101 --Propagation"
                 }
    
    def __del__(self):
        self.close()
    
    def connectInpaint(self,hostname='inpaint', username='appuser', password='appuser'):
        self.connect(hostname,username=username,password=password)
        self.isConnected = True

    def executeCommandsInpaint(self,commands):
        """
            Executes specified commands in container, returns results 
        """
        assert self.isConnected, "Client was not connected!"

        start = time()
        results = { 'stdin': [], 'stdout': [], 'stderr': [] }
        for cmd in commands:
            stdin, stdout, stderr = self.exec_command(cmd)  # non-blocking call
            exit_status = stdout.channel.recv_exit_status() # blocking call
            results['stdin'].append(stdin)
            results['stdout'].append(stdout)
            results['stderr'].append(stderr)

        finish = time()
        return results


    def testConnectionInpaint(self,testCommands=None,hardErrors=True):
        """
            Tests simple connectivity to the container
            to see if host is accessible, and all command paths are accessible
        """
        assert self.isConnected, "Client was not connected!"

        if testCommands is None: 
            testCommands = [ f'cd {self.c["workingDir"]} ; pwd',
                             f'ls {self.c["pythonPath"]}',
                             f'ls {self.c["scriptPath"]}',
                             f'ls {self.c["pretrainedModle"]}',
                           ] 

        start = time()
        errors = []
        for cmd in testCommands:
            stdin, stdout, stderr = self.exec_command(cmd)  # non-blocking call
            exit_status = stdout.channel.recv_exit_status() # blocking call
            if stderr:
                errors.append({'cmd': cmd, 'message': stderr})
            finish = time()

        if any(errors):
            disconnectInpaint()
            if hardErrors:
                for err in errors:
                    print(f"Error executing remote command:\n<<{err['cmd']}>>")
                    print(f"\nResult output:\n")
                    for l in err['message']:
                        print(l.strip())
                    return False

                raise Exception("Errors encountered while testing remote execution")
            else:
                return errors

        return True 


    def disconnectInpaint(self):
        self.close()
        self.isConnected = False
    
    def runInpaint(self,
                   frameDirPath, maskDirPath, 
                   commandScript=None,                # default pre-baked script will be used
                   inputHeight=512, inputWidth=1024,  # maximum size limited to 512x1024
                   CUDA_VISIBLE_DEVICES='',   # specify specific device if required, otherwise default
                   optionsString=''           # optional parameters string
                ):
        """
            'runInpaint' will execute a 'pre-baked' formula for inpainting based on the example from
            the https://github.com/nbei/Deep-Flow-Guided-Video-Inpainting definition.

        """
        assert self.isConnected, "Client was not connected!"

        cudaString = f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES}; " if CUDA_VISIBLE_DEVICES else ""
        if not optionsString:
            optionsString = self.c['optionsString']

        if commandScript is None:
            commandScript = cudaString + \
                            f"cd {self.c['workingDir']}; " + \
                            f"{self.c['pythonPath']} {self.c['scriptPath']} " + \
                            f"--frame_dir {frameDirPath} --MASK_ROOT {maskDirPath} " + \
                            f"--img_size {inputHeight} {inputWidth} " + \
                            optionsString
        
        start = time()
        stdin, stdout, stderr = self.exec_command(commandScript)  # non-blocking call
        exit_status = stdout.channel.recv_exit_status() # blocking call
        finish = time()

        return (stdin, stdout, stderr)

if __name__ == "__main__":
    pass
                             
        

            
            
        

