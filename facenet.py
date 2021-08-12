import os
from os import listdir, remove
import numpy as np
import cv2


from facenet_pytorch import MTCNN, InceptionResnetV1
import math

import torch
from PIL import Image
from copy import deepcopy


import h5py
import threading


imagefolder = 'media'


class HDF5Store(object):
    def __init__(self, datapath, dataset, shape=(1,), dtype=np.float32, compression="gzip", chunk_len=1):
        self.datapath = datapath
        self.dataset = dataset
        self.shape = shape
        self.vecdim = 512
        # Special
        dtype = np.dtype([('Name', 'S32'), ('Vector', np.float32, (self.vecdim,)), ('Valid', 'i')])
        self.shape = (1,)
        self.dtype = dtype
        self.inh5 = set()
        self.min_name = None
        self.min_dist = 1000
        self.min_val = 0
        self.g = 0
        self.compression = compression
        self.chunk_len = chunk_len

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Running on device: {}'.format(self.device))
        self.mtcnn = MTCNN(image_size=160, margin=0, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').to(self.device).eval()

        if self.datapath not in list(os.listdir()):

            with h5py.File(self.datapath, mode='w') as h5f:
                self.dset = h5f.create_dataset(
                    self.dataset,
                    shape=(0, ) + self.shape,
                    maxshape=(None, ) + self.shape,
                    dtype=self.dtype,
                    compression=self.compression,
                    chunks=(self.chunk_len, ) + self.shape)
                h5f.flush()
                h5f.close()
            self.i = 0

        self.refreshveclib()

    def refreshveclib(self):
        with h5py.File(self.datapath, mode='r') as h5f:
            self.veclib = np.array(list(map(self.l2_normalize, h5f[self.dataset]['Vector'].reshape(-1, self.vecdim))))
            self.i = len(self.veclib)
            # print("New:",h5f[self.dataset]['Name'],self.i)

    #     self.scandb()

    # def scandb(self):

    #     with h5py.File(self.datapath, mode='r') as h5f:
    #         self.i = h5f[self.dataset].shape[0]

    #         self.inh5 = set(map(lambda x:x.decode("utf-8") ,h5f[self.dataset]['Name'].flatten()))
    #         h5f.close()

    #     self.checknewold()

    #     with h5py.File(self.datapath, mode='r') as h5f:
    #         self.veclib = np.array(list(map(self.l2_normalize,h5f[self.dataset]['Vector'].reshape(-1,self.vecdim))))
    #         self.lendataset = len(self.veclib)

    # def checknewold(self):
    #     unfound_face_list = []
    #     listdir = os.listdir(imagefolder)
    #     setlistdir = set(listdir)
    #     inh5 = self.inh5

    #     tobeadded = setlistdir-inh5
    #     toberemoved = inh5 - setlistdir - {'Unknown'}
    #     # print("tobeadded ",tobeadded)
    #     # print("toberemoved",toberemoved)

    #     for i in tobeadded:
    #         print("\rAdding: {}".format(i),end='')

    #         fin_path = os.path.join(imagefolder,i)

    #         for imgwisepath in os.listdir(fin_path):

    #             try:
    #                 i_path = os.path.join(fin_path,imgwisepath)
    #                 img = Image.open(i_path)
    #                 img_cropped = self.mtcnn(img)

    #                 representation = self.resnet(img_cropped.unsqueeze(0).to(self.device))[0].detach().cpu().numpy()
    #                 self.append(np.array([(i,representation,1)],dtype=self.dtype))
    #             except Exception as e:
    #                 unfound_face_list.append(i)
    #                 continue

    #     if len(unfound_face_list)!=0:
    #     		print("\n\nCouldnt find faces in the following images")
    #     for x in unfound_face_list:
    #         print("\t{}".format(x))
    #     if len(unfound_face_list)>0:
    #         print("---------")
    #     for i in toberemoved:
    #         print("\rRemoving: {}".format(i),end='')
    #         self.remove(i)
    #     if len(toberemoved)>0:
    #         print("---------")
    #     del listdir
    #     del unfound_face_list
    #     del inh5
    #     del setlistdir
    #     del tobeadded
    #     del toberemoved

    def addtodb(self, name, path=None, pilimg=None):
        if path is not None:
            pilimg = Image.open(path)

        img_cropped = self.mtcnn(pilimg)
        try:
            representation = self.resnet(img_cropped.unsqueeze(0).to(self.device))[0].detach().cpu().numpy()
            assert len(name.split('-')) == 3
            self.append(np.array([(name, representation, 1)], dtype=self.dtype))
            self.refreshveclib()
            return 1
        except Exception as e:
            return -1

    def removefromdb(self, name):
        assert len(name.split('-')) == 3
        try:
            self.remove(name)
            self.refreshveclib()
            return 1
        except Exception as e:
            return -1

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
        with h5py.File(self.datapath, mode='r+') as h5f:
            count = len(h5f[self.dataset][h5f[self.dataset]['Name'] == name.encode('UTF-8'), 'Valid'])
            h5f[self.dataset][h5f[self.dataset]['Name'] == name.encode(
                'UTF-8'), 'Valid'] = np.array([0]*count, dtype=np.int32)
            h5f[self.dataset][h5f[self.dataset]['Name'] == name.encode(
                'UTF-8'), 'Vector'] = np.array(([0.0001]*self.vecdim*count), dtype=np.float32).reshape(count, -1, 512)
            h5f[self.dataset][h5f[self.dataset]['Name'] == name.encode(
                'UTF-8'), 'Name'] = np.array([b'Unknown']*count, dtype='|S32')
            h5f.flush()
            h5f.close()

    def l2_normalize(self, x):
        return x / np.sqrt(np.sum(np.multiply(x, x)))

    def findEuclideanDistance(self, source_representation, test_representation):
        euclidean_distance = source_representation - test_representation
        euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
        euclidean_distance = np.sqrt(euclidean_distance)
        return euclidean_distance

    def findnearest(self, capvec):
        name = None
        dist = 1.5
        with h5py.File(self.datapath, mode='r') as h5f:
            for i in h5f[self.dataset]:
                i = i[0]
                ittervec = i[1]
                distc = self.findEuclideanDistance(self.l2_normalize(capvec), self.l2_normalize(ittervec))
                if dist > distc and i[2] == 1:
                    dist = distc
                    name = i[0]

        return name.decode("utf-8"), dist

    def findnearest2(self, capvec):
        fin = np.linalg.norm(self.veclib-self.l2_normalize(capvec), axis=1)
        min = np.argmin(fin)

        if fin[min] < 1:
            with h5py.File(self.datapath, mode='r') as h5f:
                name = h5f['vecs']['Name'][min][0]
            return name.decode("utf-8"), fin[min]
        else:
            return None, 0

    def setminnamedist(self, name, dist, val):
        self.min_name = name
        self.min_dist = dist
        self.min_val = val

    def findnearestt(self, capvec, start, end):
        fin = np.linalg.norm(self.veclib[start:end]-self.l2_normalize(capvec), axis=1)
        min = np.argmin(fin)
        # print(fin)
        # print("self.min_dist:",self.min_dist,"fin[min]:",fin[min])
        with h5py.File(self.datapath, mode='r') as h5f:
            if fin[min] < self.min_dist:  # and h5f['vecs']['Valid'][start+min][0]!=0:
                name = h5f['vecs']['Name'][start+min][0]
                val = 1  # h5f['vecs']['Valid'][start+min][0]#h5f[self.dataset][h5f[self.dataset]['Name'][:]==name,0,'Valid']
                self.setminnamedist(name.decode("utf-8"), fin[min], val)

    def multithreadedsearch(self, capvec):
        if self.i == 0:
            return None, 0

        self.g += 1
        # print(self.g)

        threadlist = []
        x = self.i
        i = 0
        for i in range(min(100, x//100)):
            t = threading.Thread(target=self.findnearestt, args=(capvec, x//100*i, x//100*(i+1)))

            threadlist.append(t)

        t = threading.Thread(target=self.findnearestt, args=(capvec, x//100*(i+1), x))

        threadlist.append(t)

        for t in threadlist:
            t.start()
        for t in threadlist:
            t.join()

        min_name, min_dist = self.min_name, self.min_dist

        self.min_name, self.min_dist = None, 1000

        # print(min_name,self.min_val)

        if min_dist < 1 and self.min_val:
            return min_name, min_dist
        else:
            return None, 0

    def getframedetails(self, frame):

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_cropped = Image.fromarray(frame)
        # img_cropped = self.mtcnn(img_cropped).to(self.device)
        # img_cropped = torch.Tensor(img_cropped).to(self.device)
        img_cropped = self.mtcnn(img_cropped).to(self.device)
        img_embedding = self.resnet(img_cropped.unsqueeze(0))[0].detach().cpu().numpy()
        name, dist = self.multithreadedsearch(capvec=img_embedding)

        return name, dist

    def getname(self, path):

        img = cv2.imread(path, 1)
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        framePIL = Image.fromarray(np.array(frame))
        faces, _ = self.mtcnn.detect(framePIL)

        try:
            if faces == None:
                print('No face')
        except:
            for (xmin, ymin, xmax, ymax) in faces:

                xmin = int(xmin)
                xmax = int(xmax)
                ymin = int(ymin)
                ymax = int(ymax)

                w = xmax-xmin
                h = ymax-ymin
                x = xmin
                y = ymin

                if xmax-xmin > 10:  # discard small detected faces
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (67, 67, 67), 1)  # draw rectangle to main image
                    detected_face = np.array(img[int(ymin):int(ymax), int(xmin):int(xmax)])  # crop detected face

                    # detected_face = cv2.resize(detected_face, target_size) #resize to 152x152

                    # img_pixels = image.img_to_array(detected_face)
                    # img_pixels = np.expand_dims(img_pixels, axis = 0)
                    # img_pixels /= 255

                    employee_name = None
                    similarity = 0

                    try:
                        employee_name, similarity = self.getframedetails(detected_face)
                        employee_name = employee_name.split('-')[1]
                        return employee_name
                    except Exception as e:
                        return None


# if __name__ == "__main__":
#     vech5 = HDF5Store('embeddingVec.h5', 'vecs',)
#     name = vech5.getname('IMG_20210118_000604-removebg-preview.png')
#     print(name)
#     if name is None:
#         a = vech5.addtodb(name='1-Dwayne-1', path='IMG_20210118_000604-removebg-preview.png')

#     name = vech5.getname('1620934536774.jpg')
#     print(name)
#     if name is None:
#         a = vech5.addtodb(name='2-Aaqib-1', path='1620934536774.jpg')
