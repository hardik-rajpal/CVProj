import cv2
import numpy as np
class PPN:
    #puzzle piece node
    def __init__(self,piece:cv2.Mat):
        self.mat = cv2.cvtColor(piece,cv2.COLOR_BGR2GRAY)
        self.state = 0
        self.legalmoves = list(range(9))#0 to 8
    def getxy(self):
        return (self.state)%3,int((self.state)/3)
    def getindex(self,x,y):
        return x + 3*y
    def compareNodeTo(self,piece):
        sxy = self.getxy()
        pxy = piece.getxy()
        v1 = ...
        v2 = ...
        k = 3
        if(pxy[0]==sxy[0]):
            #xcoords are same=>compare rows at start and end.
            if(pxy[1]>sxy[1]):
                v1 = piece.mat[(k-1),:]
                v2 = self.mat[-(k),:]
            else:
                v1 = piece.mat[-(k),:]
                v2 = self.mat[k-1,:]
        else:
            #ycoords are same=>compare cols at end and start
            if(pxy[0]<sxy[0]):
                v1 = piece.mat[:,-(k)]
                v2 = self.mat[:,k-1]
            else:
                v1 = piece.mat[:,k-1]
                v2 = self.mat[:,-(k)]
        v1 = v1.flatten();v2 = v2.flatten()
        diff = cv2.subtract(v1,v2,dtype=cv2.CV_32F)
        # print(v1[:10],v2[:10],diff.flatten()[:10],sep='\n')
        dist = np.linalg.norm(diff,ord=2)
        dist /= (np.sqrt(v2.size))
        # print(sxy,pxy,dist)
        return 1/(1+dist)
    def vectorprox(v1,v2):
        v1 = v1.flatten();v2 = v2.flatten()
        diff = cv2.subtract(v1,v2,dtype=cv2.CV_32F)
        # print(v1[:10],v2[:10],diff.flatten()[:10],sep='\n')
        dist = np.linalg.norm(diff,ord=2)
        dist /= (np.sqrt(v2.size))
        # print(sxy,pxy,dist)
        return 1/(1+dist)
    def newCompare(self,piece,k=1):
        mat1 = self.mat
        mat2 = piece.mat
        tblr1 = [mat1[k-1,:],mat1[-k,:],mat1[:,k-1],mat1[:,-k]]
        tblr2 = [mat2[k-1,:],mat2[-k,:],mat2[:,k-1],mat2[:,-k]]
        return [
            PPN.vectorprox(tblr1[0],tblr2[1]),
            PPN.vectorprox(tblr1[1],tblr2[0]),
            PPN.vectorprox(tblr1[2],tblr2[3]),
            PPN.vectorprox(tblr1[3],tblr2[2])
            ]
    def value(self,pieces):
        xyvals = [piece.getxy() for piece in pieces]
        sxy = self.getxy()
        score = 0
        numadj = 0
        for i,xyv in enumerate(xyvals):
            if(abs(xyv[0]-sxy[0])+abs(xyv[1]-sxy[1])==1):
                # piece is adjacent
                score+=self.compareNodeTo(pieces[i])
                numadj+=1
        if(numadj==0):
            return 0
        return score/numadj