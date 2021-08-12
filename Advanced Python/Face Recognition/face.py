# from facenet_pytorch import MTCNN, InceptionResnetV1
# import math
# import cv2
# import numpy
# from PIL import Image
# import datetime
# import torch
# import os
# import pandas as pd
# from tkinter import *
# from PIL import ImageTk, Image
# from copy import deepcopy
# import numpy as np 
# import urllib, json
# from CONFIG import *



# in_dict_len=0
# media_dir='./media'

# def init_dict(media_dir=media_dir,mtcnn=None,resnet=None):
#   print("Initializing")
#   device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#   print('Running on device: {}'.format(device))

#   if mtcnn==None:
#     mtcnn = MTCNN(image_size=160, margin=0, device=device)

#   if resnet==None:
#     resnet = InceptionResnetV1(pretrained='vggface2').to(device).eval()

#   embedding_dict = {}

#   for i,a in enumerate(os.listdir(media_dir)):
#     print("\rInitialization Number - {}".format(i),end="")
    
#     try:
#       img_name_main = os.listdir(os.path.join(media_dir,a))[0]
#       srcm = os.path.join(media_dir,a,img_name_main) 
#       img = Image.open(srcm)
#       img_cropped = mtcnn(img)
#       img_embedding = resnet(img_cropped.unsqueeze(0).to(device))[0].detach().cpu().numpy()
#       embedding_dict[a]=img_embedding
#     except Exception as e:
      
#       pass

#   global in_dict_len
#   in_dict_len = len(os.listdir(media_dir))
#   print("")


#   return embedding_dict,mtcnn,resnet,device



# import mmap
# from posix_ipc import Semaphore, O_CREX, ExistentialError, O_CREAT, SharedMemory, unlink_shared_memory, O_TRUNC,O_EXCL
# from ctypes import sizeof, memmove, addressof, create_string_buffer
# from structures import MD

# md_buf = create_string_buffer(sizeof(MD))


# class ShmWrite:
#     def __init__(self, name):
#         self.shm_region = None

#         self.md_region = SharedMemory(name + '-meta', O_CREAT, size=sizeof(MD))
#         self.md_buf = mmap.mmap(self.md_region.fd, self.md_region.size)
#         self.md_region.close_fd()

#         self.shm_buf = None
#         self.shm_name = name
#         self.count = 0

#         try:
#             self.sem = Semaphore(name, O_CREX)
#         except ExistentialError:
#             sem = Semaphore(name, O_CREAT)
#             sem.unlink()
#             self.sem = Semaphore(name, O_CREX)
#         self.sem.release()

#     def add(self, frame: np.ndarray):
#         byte_size = frame.nbytes
#         if not self.shm_region:
#             self.shm_region = SharedMemory(self.shm_name, O_CREAT, size=byte_size)
#             self.shm_buf = mmap.mmap(self.shm_region.fd, byte_size)
#             self.shm_region.close_fd()

#         self.count += 1
#         md = MD(frame.shape[0], frame.shape[1], frame.shape[2], byte_size, self.count)
#         self.sem.acquire()
#         memmove(md_buf, addressof(md), sizeof(md))
#         self.md_buf[:] = bytes(md_buf)
#         self.shm_buf[:] = frame.tobytes()
#         self.sem.release()

#     def release(self):
#         self.sem.acquire()

#         self.md_buf.close()
#         unlink_shared_memory(self.shm_name + '-meta')

#         self.shm_buf.close()
#         unlink_shared_memory(self.shm_name)

#         self.sem.release()
#         self.sem.close()




# class FRAS:
  
#   def __init__(self,source,threshold,embedding_dict,mtcnn,resnet,device,api=None):
#     self.source = source
#     self.threshold = threshold
#     self.embedding_dict = embedding_dict
#     self.vote_dict = {}
#     self.i = 0
#     self.font = cv2.FONT_HERSHEY_SIMPLEX 
#     self.fontScale = 1
#     self.color = (255, 0, 0) 
#     self.thickness = 2
#     self.mtcnn = mtcnn
#     self.resnet = resnet
#     self.device = device
#     self.cap = cv2.VideoCapture(source)
#     self.api = api

#   def get_coordinates(self,coords,pad):
#     x1,y1 = coords[0]-2*pad,coords[1]-pad
#     x2,y2 = coords[2]+pad,coords[3]+pad
#     return [x1,y1,x2,y2]

#   def find_name(self,img_cropped,embedding_dict=None):

#     if embedding_dict is None:
#       embedding_dict=self.embedding_dict

#     img_cropped = Image.fromarray(img_cropped)
#     img_cropped = self.mtcnn(img_cropped).to(self.device)
#     img_embedding = self.resnet(img_cropped.unsqueeze(0))[0].detach().cpu().numpy()
#     final_name = None

#     # max = 2*math.sqrt(512)
#     max = 1
#     for name,emb in embedding_dict.items():
#       dist = numpy.linalg.norm(img_embedding-emb)
#       if dist<max:
#         final_name = name
#         max=dist

#     return final_name,dist


#   def vote(self,name,vote_dict=None,threshold=None):

#     if vote_dict is None:
#       vote_dict=self.vote_dict

#     if threshold is None:
#       threshold=self.threshold

#     if name not in list(vote_dict.keys()):
#       vote_dict[name] = {
#                           'vote':1,
#                           'time': datetime.datetime.now(),
#                         }
#     else:
#       vote_dict[name]['vote']+=1
      
    
#     if vote_dict[name]['vote']>threshold:
#       self.register(name,vote_dict[name]['time'])
#       del vote_dict[name]

#     return vote_dict

#   def register(self,name,time):

#     if self.api is None:
#       if "register.csv" not in os.listdir():
#         f = open("register.csv","w+")
#         f.write("NAME,TIME")
#         f.close()

#       df = pd.read_csv("register.csv",index_col=None,usecols=['NAME', 'TIME'])
#       df.columns = ['NAME','TIME']
#       df = df.append({'':'','NAME':name,'TIME':time},ignore_index=True)
#       df.to_csv("register.csv")

#     #Attendance API here
#     else:
#       url = self.api
#       response = urllib.urlopen(url)   
    

#   def stream(self):    

#     try:
#       cap = self.cap
#       _,frame = cap.read()
#       frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#       framePIL = Image.fromarray(numpy.array(frame))
#       boxes, _ = self.mtcnn.detect(framePIL)
#       frameCopy = deepcopy(frame)
#       try:
#         for box in boxes:
#             img = frameCopy[int(box[1]):int(box[3]),int(box[0]):int(box[2]),:]
#             name,dist = self.find_name(img)
#             vote_dict = self.vote(name)
#             if name==None:
#               self.color = (255, 0, 0) 
#             else:
#               self.color = (0, 255, 0)
            
#             box = self.get_coordinates(box,20)
#             frame = cv2.rectangle(frame, (int(box[0]),int(box[1])),(int(box[2]),int(box[3])) , self.color, self.thickness)
#             frame = cv2.putText(frame, str(name), (int(box[0]),int(box[1])), self.font,  
#                       self.fontScale, self.color, self.thickness, cv2.LINE_AA)
#       except Exception as e:
#         # print(e)
#         pass

#       try:
#         print('\rTracking frame: {} | Number of faces: {}'.format(self.i + 1,len(boxes)), end='')
#       except:
#         print('\rTracking frame: {} | Number of faces: {}'.format(self.i + 1,0), end='')
          
#     except Exception as e:
#       pass

#     self.i+=1
    
#     frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#     return frame



# def concat_vh(list_2d): 
  
#     return cv2.vconcat([cv2.hconcat(list_h)  
#                         for list_h in list_2d]) 

# def stack(frame_lst,h):

#   frame_lst = [cv2.resize(x,(500,400)) for x in frame_lst]

#   l = len(frame_lst)
#   if l%h!=0:
#     rem = h - len(frame_lst)%h
#     for _ in range(rem): 
#       frame_lst.append(frame_lst[0]*0)  


#   frame_lst = np.array(frame_lst)

#   frame_lst = frame_lst.reshape(-1,h,400,500,3)

#   return concat_vh(frame_lst)

# def face(source1=0,source2=2):

#   emb_dict,mtcnn,resnet,device = init_dict()

#   stream1 = FRAS(source1,20,emb_dict,mtcnn,resnet,device,api=SOURCE_1_API)
#   stream2 = FRAS(source2,20,emb_dict,mtcnn,resnet,device,api=SOURCE_2_API)
    
#   while True:  

#     try:
#       shm_w = ShmWrite('abc')
#       frame1 = stream1.stream()
#       frame2 = stream2.stream()

#       fframe = stack([frame1,frame2],2)

#       if len(os.listdir(media_dir))!=in_dict_len:
#         emb_dict,mtcnn,resnet,device = init_dict(mtcnn=mtcnn,resnet=resnet)

#       # fframe = stack([frame1],1)
#       # cv2.imshow("frame",fframe)
#       # k = cv2.waitKey(10)

#       shm_w.add(fframe)

#     except KeyboardInterrupt:
#       print("Exiting")
#       # shm_w.release()
#       del mtcnn
#       del resnet
#       break

#     # if cv2.getWindowProperty('frame',cv2.WND_PROP_VISIBLE) < 1:        
#     #   break  
#     # if k==27:
#     #   break
    
#   cv2.destroyAllWindows()    
  
# if __name__=="__main__":
#   face(SOURCE_1,SOURCE_2)



# # import face from face

# # face("rtsp://1898407539874","rtsp://763597642975")
