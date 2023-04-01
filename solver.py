import argparse,os,cv2,random
from ds import PPN
import numpy as np
def showSmt(segppns,shuffled_segppns):
    print('unshuffled: ',end=' ')
    unshuffledvals = [];shuffledvals = []
    for p in segppns:
        unshuffledvals.append(round(p.value(segppns),2))
        print(unshuffledvals[-1],end=', ')
    print('avg: ',np.round(np.average(unshuffledvals),2))
    print('\nshuffled: ',end=' ')
    for p in shuffled_segppns:
        shuffledvals.append(round(p.value(shuffled_segppns),2))
        cv2.imshow(f'{p.state}',p.mat)
        print(shuffledvals[-1],end=', ')
    cv2.waitKey(0)
    print('avg: ',np.round(np.average(shuffledvals),2))
    print()

if __name__=='__main__':
    random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',type=str)
    args = parser.parse_args()
    segmentspath = args.path
    #ex: segmented/general/0/
    if(not segmentspath.endswith('/')):segmentspath+='/'
    segbasenames = os.listdir(segmentspath)
    segpaths = list(map(lambda x:f'{segmentspath}/{x}',segbasenames))
    segmats = list(map(lambda path:cv2.imread(path),segpaths))
    segppns = list(map(lambda segmat:PPN(segmat),segmats))
    shuffled_segmats = segmats.copy()
    random.shuffle(shuffled_segmats)
    shuffled_segppns = list(map(lambda segmat: PPN(segmat),shuffled_segmats))
    for i in range(len(segppns)):
        segppns[i].state = i
    for i in range(len(shuffled_segppns)):
        shuffled_segppns[i].state = i
    # showSmt(segppns,shuffled_segppns)
    corrs = np.zeros((9,9,4))
    for k in range(0,5):
        for i in range(len(segppns)):
            for j in range(len(segppns)):
                corrs[i,j,:] = np.array(segppns[i].newCompare(segppns[j],k))
        cv2.imshow('corrs',np.max(corrs,2))
        artcorrs = np.zeros((9,9))
        for i in range(9):
            for j in range(9):
                if(i!=j):
                    if(abs((i%3)-(j%3)) + abs(int(i/3)-int(j/3)) == 1):
                        artcorrs[i,j] = 1
        cv2.imshow('artcorrs',artcorrs)
        cv2.waitKey(0)