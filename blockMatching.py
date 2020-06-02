import numpy as np
import typing
import pdb
import pywt
import matplotlib.pyplot as plt
#%% Block matching

def exhaustiveSearch(image: np.array, n1: int, n2: int,stride: int)-> np.array:
    '''
    Desc.:
        Implements the block matching algorithm through an exhaustive search
    ----------
    
    
    Parameters
    ----------
    image:
        DESCRIPTION: noisy image - currently only grayscale images are supported
    n1:
        DESCRIPTION: patch size; said to be "large enough" in 
        https://hcburger.com/files/neuraldenoising.pdf (they use 17*17) patches
        set to 10 in our paper
    n2:
        DESCRIPTION: dim 2 of the stacks; set to 32 in our paper
    stride:
        DESCRIPTION: stride between patches. Set it to one for best results, 
        but computationally expansive. stride = 3 is said to have good results cf :
        https://hcburger.com/files/neuraldenoising.pdf sec. 3.2

    Returns
    -------
    
    list of image stacks

    '''
    l, c = image.shape
    
    # -- Getting patch indexes as tuples and then corresponding patches
    patchMap = np.zeros((l,c))
    patchLength = int(n1/2)
    patchMap[patchLength:l-patchLength:stride,patchLength:c-patchLength:stride] = 1
    patchMap = np.where(patchMap)
    patchIndexes = [(patchMap[0][i],patchMap[1][i]) for i in range(len(patchMap[0]))]
    patches = [patchFromIndex(image,patchIndexes[i],n1) for i in range(len(patchIndexes))]
    
    # -- looking for similar patches
    dissimilarityMatrix = computeDissimilarityMatrix(patches)
    
    # -- Retrieving stacks
    stacks = retrieveStacks(patches, dissimilarityMatrix, n2)
    return stacks

    
    


def patchFromIndex(image: np.array, idx: tuple, n1: int)-> np.array:
    '''

    Returns
    -------
    patch of size n1*n1

    '''
    patchLength = int(n1/2)
    line, col = idx[0], idx[1]
    return image[line-patchLength:line +patchLength+1, col - patchLength: col + patchLength+1] 
    
    
    
    
    
def dissimilarity(p: np.array, q: np.array) -> float:
    '''
    
    Parameters
    ----------
    p:
        DESCRIPTION: current patch
    q:
        DESCRIPTION : tested patch

    Returns
    -------
    float
        DESCRIPTION : dissimilarity between these patches

    '''
    return np.linalg.norm(p-q)

def computeDissimilarityMatrix(patches :list) -> np.array:
    '''


    Returns
    -------
    Array of size len(patches)*len(patches)
        DESCRIPTION ; element i,j from the array is the dissimilarity between patch i and j from the patches

    '''
    size = len(patches)
    dissimilarityMatrix = np.zeros((size,size))
    print('Computing dissimilarity matrix : ')
    for k in range(size-1):
        for l in range(k+1,size):
            dissimilarityMatrix[k,l] = dissimilarity(patches[k],patches[l])
            if k % int(size/10) ==0 and l == k+1:
                print('.',end = '')
            
    dissimilarityMatrix += dissimilarityMatrix.T
    
    # -- Setting elements from diagonal to infinity so that they're not seen as similar to themselves
    np.fill_diagonal(dissimilarityMatrix, float('inf'))
    
    return dissimilarityMatrix

def retrieveStacks(patches: list, dissimilarityMatrix: np.array, n2:int) -> list:
    '''
    Description
    ----------
    
        Looks for patches that have not been put to a stack yet, retrieve n2 of their 
        most similar patches and put them to a new stack
    Parameters
    ----------
    list : patches
        DESCRIPTION.
    np.array : dissimilarityMatrix
        DESCRIPTION.

    Returns
    -------
    list of the stacks

    '''
    # -- Local change of patches to easy computations
    patches = np.array(patches)
    
    # -- Retrieve index of n2 most similar patches, current patch excluded
    similarPatchesIndexes = dissimilarityMatrix.argsort(axis = 0)[:n2-1] 
    
    # -- stackCount gives how many times a given patch has been put into a stack
    stackCount = [0]*len(patches)
    stacks = []
    
    while True:
        try :
            # -- Update of the stack; this part could be optimized
            patchIndex = stackCount.index(0)
            patch = patches[patchIndex]
            similarPatches = list(patches[similarPatchesIndexes[:,patchIndex]])
            similarPatches.append(patch) 
            similarPatches = np.transpose(np.array(similarPatches), (1,2,0))            
            stacks.append(similarPatches) 
            
            # -- Update of stackCount; this part could be optimized too
            stackCount = np.array(stackCount)
            stackCount[patchIndex] +=1
            stackCount[similarPatchesIndexes[:,patchIndex]]+=1
            stackCount = list(stackCount)
            
        except ValueError: # no more zeros in stackCount
            break
        
    return stacks

def showStacksFromImage(imagePath: str,n1: int, n2: int, strides: int):
    '''
    Desc : Shows the stacks in the iPython console
    For the moment works only if imagePath = 'lena.png'
    we used it with n1 : 17 n2: 32 strides :3 -> yields good results and is quick
    '''
    image = plt.imread(imagePath)
    image = np.delete(image,3,2).mean(axis = 2)[200:250,100:200] #converting to grayscale and extracting
    stacks = exhaustiveSearch(image, n1, n2, strides)
    print('Part of the image used for computations:')
    plt.imshow(image,cmap = 'gray')
    print('\n Which stack do you want to see (int from 0 to %i):'%(len(stacks)-1), end= '')
    idx = int(input())
    stack = stacks[idx]
    
    fig,axes = plt.subplots(2,2)
    fig.suptitle('Most similar blocks of the stack %i'%idx)
    axes[0,0].imshow(stack[:,:,0],cmap = 'gray')
    axes[0,0].axis('off')
    axes[0,1].imshow(stack[:,:,1],cmap = 'gray')
    axes[0,1].axis('off')
    axes[1,0].imshow(stack[:,:,2],cmap = 'gray')
    axes[1,0].axis('off')
    axes[1,1].imshow(stack[:,:,3],cmap = 'gray')
    axes[1,1].axis('off')
    plt.show()
    return image,stack

def showStacks(stacks: np.array)->None:
    print('\n Which stack do you want to see (int from 0 to %i):'%(len(stacks)-1), end= '')
    idx = int(input())
    stack = stacks[idx]
    fig,axes = plt.subplots(2,2)
    fig.suptitle('Most similar blocks of the stack %i'%idx)
    axes[0,0].imshow(stack[:,:,0],cmap = 'gray')
    axes[0,0].axis('off')
    axes[0,1].imshow(stack[:,:,1],cmap = 'gray')
    axes[0,1].axis('off')
    axes[1,0].imshow(stack[:,:,2],cmap = 'gray')
    axes[1,0].axis('off')
    axes[1,1].imshow(stack[:,:,3],cmap = 'gray')
    axes[1,1].axis('off')
    plt.show()
    


#%% Filtering - Haar Wavelet Transform

def filterStacks(stacks :list, levels :int, tau:float)-> np.array:
    '''
    Parameters
    -------
        levels: number of levels of the wavelet transform 
        tau: regularization parameter described in the paper (not used for now)
    Returns
    -------
    stacks filtered alongside the third dimension using the Haar Wavelet transform

    '''
    # -- signals to filter; a signal is a slice of the stack in the third dimension as it is proposed in the paper
    n  = len(stacks)
    n1 = stacks[0].shape[0]
    n2 = stacks[0].shape[2]
    signalStack = [[(stacks[l][i,j,:],i,j) for i in range(n1) for j in range(n1)] for l in range(n) ] #probably not the most efficient way 
    filteredSignalStack = np.zeros((n,n1,n1,n2))
    
    # -- Parameters of the DWT
    waveFilter = pywt.Wavelet('db1')
    threshold  = np.sqrt(2*np.log10(n2*(1-2**(-levels)))) # universal threshold
    
    # -- Apply the DWT with hard threshold
    for idxStack,signals in enumerate(signalStack): 
        for idx in range(len(signals)):
            i,j = signals[idx][1], signals[idx][2]
            coeffs = pywt.wavedec(signals[idx][0], waveFilter, level=levels) # DWT coeffs
            coeffsT = gamma(coeffs, threshold) # coeffs after appli. of thr
            waveFilteredSignal = pywt.waverec(coeffsT, waveFilter, mode='per') # filtered signal
            filteredSignalStack[idxStack][i,j] = waveFilteredSignal
    
    return filteredSignalStack
        


def coeff1Dthresh(coeffs, thr, mode='hard', verbose=0):
    ''' 
    Desc:
    Applies the universal threshold to the detail sub-bands of the DWT
    put it instead of gamma
    
    Note: Borrowed from Mr Cagnazzo
    '''
    
    out = []
    out.append(coeffs[0])       
    for levelIdx in range(1,len(coeffs)):
        if verbose:
            print('Level %d '% (levelIdx))
        tmp = np.array(coeffs[levelIdx])
        tmp[np.abs(tmp)<=thr]=0
        if mode =='soft': 
            tmp[tmp>= thr] = tmp[tmp>= thr]-thr
            tmp[tmp<=-thr] = tmp[tmp<=-thr]+thr
        out.append(tmp)
    return out

def gamma(coeffs:list, tau: float)-> list:
    '''
    Desc:
        Smoother than coeff1Dthresh
    Parameters
    ----------
    coeffs : list of np.array
        DESCRIPTION: list of coefficients of the DWT by sub-band
    tau : float
        DESCRIPTION: threshold
    Returns
    -------
    regularized coefficients
    '''
    regCoeffs = []
    for q in coeffs:
        regCoeffs.append(q* (1-(tau**2/(q**2+tau**2))))
    return regCoeffs
    
    

    
                    
