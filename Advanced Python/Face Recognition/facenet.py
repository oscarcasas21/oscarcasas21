import os
from os import listdir
import numpy as np
import cv2


from facenet_pytorch import MTCNN, InceptionResnetV1
import math

import torch
from PIL import Image
from copy import deepcopy



import h5py
import threading





class HDF5Store(object):
    def __init__(self, datapath, dataset, shape=(1,), dtype=np.float32, compression="gzip", chunk_len=1):
        self.datapath = datapath
        self.dataset = dataset
        self.shape = shape
        self.vecdim = 512
        #Special 
        dtype = np.dtype([('Name', 'S32'), ('Vector', np.float32, (self.vecdim,)),('Valid','i')])
        self.shape = (1,)
        self.dtype = dtype
        self.inh5 = set()
        self.min_name = None
        self.min_dist = 1000
        self.min_val = 0
        self.g = 0



        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Running on device: {}'.format(self.device))
        self.mtcnn = MTCNN(image_size=160, margin=0, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').to(self.device).eval()








        if self.datapath not in list(os.listdir()):

            with h5py.File(self.datapath, mode='w') as h5f:
                self.dset = h5f.create_dataset(
                    dataset,
                    shape=(0, ) + shape,
                    maxshape=(None, ) + shape,
                    dtype=dtype,
                    compression=compression,
                    chunks=(chunk_len, ) + shape)
                h5f.flush()
                h5f.close()
            self.i = 0
        else:
            with h5py.File(self.datapath, mode='r') as h5f:
                self.i = h5f[dataset].shape[0]
                # print('/////////',self.i)
                self.inh5 = set(map(lambda x:x.decode("utf-8") ,h5f[dataset]['Name'].flatten()))
                h5f.close()
                
        self.checknewold()

        with h5py.File(self.datapath, mode='r') as h5f:
            self.veclib = np.array(list(map(self.l2_normalize,h5f[self.dataset]['Vector'].reshape(-1,self.vecdim))))
            self.lendataset = len(self.veclib)
            print('List of current images in the database:')
            for x,y in zip(h5f['vecs']['Name'],h5f['vecs']['Valid']):
                print(x[0],y[0])
            print("-------------")



    def checknewold(self):
        unfound_face_list = []
        listdir = os.listdir('images')
        setlistdir = set(listdir)
        inh5 = self.inh5

        tobeadded = setlistdir-inh5
        toberemoved = inh5 - setlistdir - {'Unknown'}
        # print("tobeadded ",tobeadded)
        # print("toberemoved",toberemoved)
        for i in tobeadded:
            try: 
                print("\rAdding: {}".format(i),end='')

                # try:
                #     with h5py.File(self.datapath, mode='r+') as h5f:
                #         if len(h5f['vecs'][h5f['vecs']['Name']==i,'Valid'])==1:
                #             h5f[self.dataset][h5f[self.dataset]['Name']==i,'Valid'] = np.array([1],dtype=np.int32)
                #             continue
                # except Exception as e:
                #     print(e)
                #     pass

                fin_path = os.path.join('images',i)
                
                img = Image.open(fin_path)
                img_cropped = self.mtcnn(img)

                representation = self.resnet(img_cropped.unsqueeze(0).to(self.device))[0].detach().cpu().numpy()
                    
            except Exception as e:
                print(e)
                unfound_face_list.append(i)
                continue



            self.append(np.array([(i,representation,1)],dtype=self.dtype))
        if len(unfound_face_list)!=0:
        		print("\n\nCouldnt find faces in the following images")
        for x in unfound_face_list:
            print("\t{}".format(x))
        print("---------")
        for i in toberemoved:
            print("\rRemoving: {}".format(i),end='')
            self.remove(i)
        print("---------")
        del listdir
        del unfound_face_list
        del inh5
        del setlistdir
        del tobeadded
        del toberemoved

    def append(self, values):
        with h5py.File(self.datapath, mode='a') as h5f:
            dset = h5f[self.dataset]
            dset.resize((self.i + 1, ) + self.shape)
            dset[self.i] = [values]
            self.i += 1
            # print(values[0][0])
            # print(h5f[self.dataset][h5f[self.dataset]['Name'][:]==values[0][0],0,'Valid'])
            h5f.flush()
            h5f.close()
            
    # def remove(self, name):
    #     # print('/////////',name)
    #     with h5py.File(self.datapath, mode='r+') as h5f:
    #         print(len(h5f[self.dataset][h5f[self.dataset]['Name']==name.encode('UTF-8'),'Valid']))
    #         h5f[self.dataset][h5f[self.dataset]['Name']==name.encode('UTF-8'),'Valid'] = np.array([0],dtype=np.int32)
    #         h5f[self.dataset][h5f[self.dataset]['Name']==name.encode('UTF-8'),'Vector'] = np.array([0.0001]*self.vecdim,dtype=np.float32)
    #         h5f[self.dataset][h5f[self.dataset]['Name']==name.encode('UTF-8'),'Name'] = np.array([b'UNKNOWN'],dtype='|S32')
    #         # h5f[][h5f[self.dataset]['Name'][:]==name,0,'Valid'] = 0    
    #         h5f.flush()
    #         h5f.close()

    def remove(self, name):
        # print('/////////',name)
        with h5py.File(self.datapath, mode='r+') as h5f:
            count = len(h5f[self.dataset][h5f[self.dataset]['Name']==name.encode('UTF-8'),'Valid'])
            print("len: ",count)
            h5f[self.dataset][h5f[self.dataset]['Name']==name.encode('UTF-8'),'Valid'] = np.array([0]*count,dtype=np.int32)
            h5f[self.dataset][h5f[self.dataset]['Name']==name.encode('UTF-8'),'Vector'] = np.array(([0.0001]*self.vecdim*count),dtype=np.float32).reshape(count,-1,512)
            h5f[self.dataset][h5f[self.dataset]['Name']==name.encode('UTF-8'),'Name'] = np.array([b'Unknown']*count,dtype='|S32')
            # h5f[][h5f[self.dataset]['Name'][:]==name,0,'Valid'] = 0    
            h5f.flush()
            h5f.close()


    def l2_normalize(self,x):
        return x / np.sqrt(np.sum(np.multiply(x, x)))

    def findEuclideanDistance(self,source_representation, test_representation):
        euclidean_distance = source_representation - test_representation
        euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
        euclidean_distance = np.sqrt(euclidean_distance)
        return euclidean_distance


    def findnearest(self,capvec):
        name = None
        dist = 1.5
        with h5py.File(self.datapath, mode='r') as h5f:
            for i in h5f[self.dataset]:
                i = i[0]
                ittervec = i[1]
                distc = self.findEuclideanDistance(self.l2_normalize(capvec), self.l2_normalize(ittervec))
                if dist>distc and i[2]==1:
                    dist = distc
                    name = i[0]

        return name.decode("utf-8"),dist

    def findnearest2(self,capvec):
        fin = np.linalg.norm(self.veclib-self.l2_normalize(capvec),axis=1)
        min = np.argmin(fin)

        if fin[min]<1:
            with h5py.File(self.datapath, mode='r') as h5f:
                name = h5f['vecs']['Name'][min][0]
            return name.decode("utf-8"),fin[min]
        else:
            return None,0

    def setminnamedist(self,name,dist,val):
        self.min_name = name
        self.min_dist = dist
        self.min_val = val

    def findnearestt(self,capvec,start,end):
        fin = np.linalg.norm(self.veclib[start:end]-self.l2_normalize(capvec),axis=1)
        min = np.argmin(fin)
        # print(fin)
        # print("self.min_dist:",self.min_dist,"fin[min]:",fin[min])
        with h5py.File(self.datapath, mode='r') as h5f:
            if fin[min]<self.min_dist:# and h5f['vecs']['Valid'][start+min][0]!=0:
                name = h5f['vecs']['Name'][start+min][0]
                print("Name:",name)
                val = 1 #h5f['vecs']['Valid'][start+min][0]#h5f[self.dataset][h5f[self.dataset]['Name'][:]==name,0,'Valid']
                self.setminnamedist(name.decode("utf-8"),fin[min],val)

    def multithreadedsearch(self,capvec):

        self.g+=1
        # print(self.g)


        threadlist = []
        x = self.lendataset
        i=0
        for i in range(min(100,x//100)):
            t = threading.Thread(target=self.findnearestt, args=(capvec,x//100*i,x//100*(i+1)))
            
            threadlist.append(t)


        t = threading.Thread(target=self.findnearestt, args=(capvec,x//100*(i+1),x))

        threadlist.append(t)
        

        for t in threadlist:
            t.start()
        for t in threadlist:
            t.join()

        min_name,min_dist = self.min_name,self.min_dist

        self.min_name,self.min_dist = None,1000


        # print(min_name,self.min_val)

        if min_dist<1 and self.min_val:
            return min_name,min_dist
        else:
            return None,0            


    def getframedetails(self,frame):

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_cropped = Image.fromarray(frame)
        # img_cropped = self.mtcnn(img_cropped).to(self.device)
        # img_cropped = torch.Tensor(img_cropped).to(self.device)
        img_cropped = self.mtcnn(img_cropped).to(self.device)
        img_embedding = self.resnet(img_cropped.unsqueeze(0))[0].detach().cpu().numpy()
        name,dist = self.multithreadedsearch(capvec=img_embedding)

        return name,dist












def streamfn():

    vech5 = HDF5Store('embeddingVec.h5','vecs',)
    cap = cv2.VideoCapture(0) #webcam

    while(True):
        ret, img = cap.read()
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        framePIL = Image.fromarray(np.array(frame))
        faces, _ = vech5.mtcnn.detect(framePIL)

        try:
            if faces == None:
                pass
        except:
            for (xmin,ymin,xmax,ymax) in faces:

                xmin = int(xmin)
                xmax = int(xmax)
                ymin = int(ymin)
                ymax = int(ymax)


                w = xmax-xmin
                h = ymax-ymin
                x = xmin
                y = ymin
                

                if xmax-xmin > 130: #discard small detected faces
                    cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (67, 67, 67), 1) #draw rectangle to main image
                    detected_face = np.array(img[int(ymin):int(ymax), int(xmin):int(xmax)]) #crop detected face
                    # detected_face = cv2.resize(detected_face, target_size) #resize to 152x152
                    
                    # img_pixels = image.img_to_array(detected_face)
                    # img_pixels = np.expand_dims(img_pixels, axis = 0)
                    # img_pixels /= 255
                    



                    employee_name = None
                    similarity = 0
                    
                    try:
                        employee_name,similarity = vech5.getframedetails(detected_face)

                    except Exception as e:
                        print(e) 
                        pass

                    if employee_name!=None:
                        path = os.path.join('media',employee_name)
                        path = os.path.join(path,os.listdir(path)[0])
                        display_img = cv2.imread(path)

                        pivot_img_size = 112
                        display_img = cv2.resize(display_img, (pivot_img_size, pivot_img_size))
                                        
                        try:
                            resolution_x = img.shape[1]; resolution_y = img.shape[0]
                            
                            label = employee_name.split(".")[0]+" ("+"{0:.2f}".format(similarity)+")"
                            
                            if y - pivot_img_size > 0 and x + w + pivot_img_size < resolution_x:
                                #top right
                                img[y - pivot_img_size:y, x+w:x+w+pivot_img_size] = display_img
                                cv2.putText(img, label, (x+w, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (67,67,67), 1)                  
                                
                                #connect face and text
                                cv2.line(img,(x+int(w/2), y), (x+3*int(w/4), y-int(pivot_img_size/2)),(67,67,67),1)
                                cv2.line(img, (x+3*int(w/4), y-int(pivot_img_size/2)), (x+w, y - int(pivot_img_size/2)), (67,67,67),1)
                            elif y + h + pivot_img_size < resolution_y and x - pivot_img_size > 0:
                                #bottom left
                                img[y+h:y+h+pivot_img_size, x-pivot_img_size:x] = display_img
                                cv2.putText(img, label, (x - pivot_img_size, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (67,67,67), 1)
                                
                                #connect face and text
                                cv2.line(img,(x+int(w/2), y+h), (x+int(w/2)-int(w/4), y+h+int(pivot_img_size/2)),(67,67,67),1)
                                cv2.line(img, (x+int(w/2)-int(w/4), y+h+int(pivot_img_size/2)), (x, y+h+int(pivot_img_size/2)), (67,67,67),1)
                                
                            elif y - pivot_img_size > 0 and x - pivot_img_size > 0:
                                #top left
                                img[y-pivot_img_size:y, x-pivot_img_size:x] = display_img
                                cv2.putText(img, label, (x - pivot_img_size, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (67,67,67), 1)
                                
                                #connect face and text
                                cv2.line(img,(x+int(w/2), y), (x+int(w/2)-int(w/4), y-int(pivot_img_size/2)),(67,67,67),1)
                                cv2.line(img, (x+int(w/2)-int(w/4), y-int(pivot_img_size/2)), (x, y - int(pivot_img_size/2)), (67,67,67),1)
                                
                            elif x+w+pivot_img_size < resolution_x and y + h + pivot_img_size < resolution_y:
                                #bottom righ
                                img[y+h:y+h+pivot_img_size, x+w:x+w+pivot_img_size] = display_img
                                cv2.putText(img, label, (x+w, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (67,67,67), 1)
                                
                                #connect face and text
                                cv2.line(img,(x+int(w/2), y+h), (x+int(w/2)+int(w/4), y+h+int(pivot_img_size/2)),(67,67,67),1)
                                cv2.line(img, (x+int(w/2)+int(w/4), y+h+int(pivot_img_size/2)), (x+w, y+h+int(pivot_img_size/2)), (67,67,67),1)
                            
                        except Exception as e:
                            print("exception occured: ", str(e))

            
        cv2.imshow('img',img)
            
        if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
            break

        if cv2.getWindowProperty('img',cv2.WND_PROP_VISIBLE) < 1:        
            break
            
    #kill open cv things        
    cap.release()
    cv2.destroyAllWindows()
    del vech5

if __name__=="__main__":
    streamfn()
