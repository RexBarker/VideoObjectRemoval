import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('inpaint', username='appuser', password='appuser') 

client
stdin, stdout, stderr = client.exec_command('ls -l')

for line in stdout:
    print(line.strip('\n'))


client.close()