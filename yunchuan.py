import os
import paramiko

#服务器信息，主机名（IP地址）、端口号、用户名及密码
hostname = "47.114.143.177"
port = 22
username = "root"
password = "Zj19870521"

class optServer:

    def __init__(self,hostname,port,username,password):
        #访问linux文件
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect(hostname, port, username, password, compress=True)
        self.sftp_client = self.client.open_sftp()
        #交互式shell
        self.channel = self.client.invoke_shell() # 在SSH server端创建一个交互式的shell
        #上传下载
        self.transport = paramiko.Transport((hostname, port))
        self.transport.connect(username = username, password = password)
        self.sftp = paramiko.SFTPClient.from_transport(self.transport)
    
    def makedirInServers(self,dirNames):
        try :
            self.sftp_client.mkdir(dirNames)
            return "{}{}{}".format("folder ",dirName," was created successfully")
        except Exception as Error:
            name = [name for name in self.sftp_client.listdir() if dirNames in name][0]
            return "{}{}".format(name,' is in here')
    
    def listFlieFromServers(self,path,fileType):
        return ["{}{}".format(path,i) for i in self.sftp_client.listdir(path) if fileType in i]
    
    def cmd(self,cmd):
        self.client.exec_command(cmd)

    def delFlie(self,filePath):
        splitpath = filePath.split('/')
        dirpath = "".join(["{0}{1}{0}".format('/',i) for i in splitpath[:-1] if i != ''])
        fileName = splitpath[-1]
        self.client.exec_command('rm'+dirpath+"rf"+fileName)
        check = sum([1 if i==fileName else 0 for i in self.sftp_client.listdir(dirpath)])
        return "{}{}{}".format('the file ',fileName,' has been deleted')

    def downloadFile(self,filePath,local_path=os.getcwd()):
        name = filePath.split('/')[-1]
        local_path = "{}{}{}".format(local_path,'/',name)
        self.sftp.get(filePath,local_path)
        return local_path
    
    def createFilePath(self,filepath):
        checkpath,checkname = self.split_opt_func(filepath) 
        create_path_cmd = "{}{}".format("mkdir ",checkpath)
        create_file_cmd = "{}{}{}{}".format("touch ",checkpath,"/",checkname)
        self.cmd(create_path_cmd)
        self.cmd(create_file_cmd)
        check_createrd = self.listFlieFromServers(checkpath,checkname)[0]
        return "{}{}{}".format("file ",check_createrd," created successfully")
    
    def checkFile(self,file_path):
        checkpath , checkname = self.split_opt_func(file_path) 
        check_file_numbers = len(self.listFlieFromServers(checkpath , checkname))
        return check_file_numbers > 0 and True or False

    def putFile(self,local_path,file_path):
        self.sftp.put(local_path,file_path)
        return self.checkFile(file_path) == True and 'file "'+file_path+'" has been uploaded successfully' or 'file "'+file_path+'" upload failed'

    def split_opt_func(self,filepath):
        splitpath = filepath.split('/')
        filename = splitpath[-1]
        checkpath = "".join(["{}{}".format("/",path) for path in splitpath[:-1]][1:])
        return checkpath,filename
        
    def __del__(self):
        self.sftp_client.close()
        self.sftp.close()
        self.client.close()
        print(hostname + " is closed")
