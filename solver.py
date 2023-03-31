import argparse,os,cv2,random
from ds import PPN
import numpy as np
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
    