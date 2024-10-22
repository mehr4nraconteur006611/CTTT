import sklearn.neighbors
import numpy as np
from timeit import default_timer as timer  
from numba.experimental import jitclass
from numba import int32, float64, float32, typed, jit, njit
from numba.extending import overload, overload_method
import torch
import scipy.spatial


def myFPFH(pointcloud):

    nneighbors = 128

    # print(pointcloud.shape)
    pa=sklearn.neighbors.KDTree(pointcloud)
    ap=pa.query(pointcloud,nneighbors+1)
    rad=ap[0][:,1].mean()
    
    et = 0.1
    # # rad = 1000000
    # print('/////////',ap[1].shape)
    # a=ap[1][:,1:]
    # print('/////////',a.shape)

    div = 4
    Icp = FPFH(et, div, nneighbors, rad,ap[1][:,1:])

    hS,normS,indS=Icp.findHistFPFH3(pointcloud) #9.5
    
    
    return  hS


# class myFPFH(object):
#     def __call__(self, pointcloud):
#         # pointcloud=pointcloud[:,:3]
#         assert len(pointcloud.shape)==2
                
# #         so1 = o3d.geometry.PointCloud()
# #         so1.points = o3d.utility.Vector3dVector(pointcloud)
        
# #         a=np.linalg.norm(np.matlib.repmat(o3d.geometry.PointCloud.get_center(so1)  ,len(pointcloud[:,1]),1)-pointcloud,axis=1)
        
        
# #         rad = a.mean()  # 0.03
# # #         print('rad 1: ',rad)
          
#         pa=sklearn.neighbors.KDTree(pointcloud)
#         ap=pa.query(pointcloud,2)
#         rad=ap[0][:,1].mean()
        
#         et = 0.1
#         # rad = 1000000
#         nneighbors = 8
#         div = 2
#         Icp = FPFH(et, div, nneighbors, rad)
    
#         hS,normS,indS=Icp.findHistFPFH3(pointcloud) #9.5
        
#         # FPFH_points=np.concatenate((pointcloud, hS), axis=1)
        
# #         print(FPFH_points.shape)
# #         print(FPFH_points)
        
# #         print(FPFH_points.shape())

        
#         return  hS
    
    

class FPFH(object):

    """Parent class for PFH"""

    def __init__(self, e, div, nneighbors, rad,ind):
        """Pass in parameters """
        self._e = e
        self._div = div
        self._nneighbors = nneighbors
        self._radius = rad
        self._ind =ind
        #self._error_list = []
        #self._Rlist = []
        #self._tlist = []


    def findHistFPFH2(self, pcS):
        """Find matches from source to target points

        :pcS: Source point cloud
        :pcT: Target point cloud
        :returns: TODO

        """
        # print("...Finding correspondences. \n")
        numS = len(pcS)

        # print("...Processing source point cloud...\n")
        normS,indS = FPFH.calc_normals2(pcS)

        # normS1,indS1 = self.calc_normals(pcS)

        ''' TODO: implement the different histograms '''
        #histS = calcHistArray_naive(pcT, normS, indS, div, nneighbors)
        #histS = calcHistArray_simple(pcT, normS, indS, div, nneighbors)
        histS = FPFH.calcHistArray2(pcS, normS, indS)

        return histS,normS,indS

    
    def findHistFPFH3(self, pcS):
        """Find matches from source to target points

        :pcS: Source point cloud
        :pcT: Target point cloud
        :returns: TODO

        """
#         pcS = np.zeros((900,3), dtype=np.float32)
#         print("...Finding correspondences. \n")
        numS = len(pcS)

        # print("...Processing source point cloud...\n")
        # normS,indS = calc_normals(self._nneighbors, self._radius, pcS)
        normS,indSs = self.calc_normals( pcS)
        indS=self._ind
        # normS1,indS1 = self.calc_normals(pcS)

        ''' TODO: implement the different histograms '''
        #histS = calcHistArray_naive(pcT, normS, indS, div, nneighbors)
        #histS = calcHistArray_simple(pcT, normS, indS, div, nneighbors)
        
        e = self._e 
        div = self._div 
        nneighbors = self._nneighbors 
        rad = self._radius 
        
        print(e)
        print(div)
        print(nneighbors)
        print(rad)
        print(len(normS))
        print(indS.shape)

        # indS = typed.List(indS)
        # normS = typed.List(normS)
        
        histS = calcHistArray2(e, div, nneighbors, rad, pcS, normS, indS)

        return histS,normS,indS


    def getNeighbors(self, pq, pc):
        """Get k nearest neighbors of the query point pq from pc, within the radius

        :pq: TODO
        :pc: TODO
        :returns: TODO

        """
        k = self._nneighbors
        neighbors = []
        for i in range(len(pc)):
            dist = np.linalg.norm(np.subtract(pq,pc[i,:]))
            if dist <= self._radius: #0.005
                neighbors.append((dist, i))
        #print("Found {} neighbors".format(len(neighbors)))
        neighbors.sort(key=lambda x:x[0])
        neighbors.pop(0)
        return neighbors[:k]
    
    # @jit(nopython=True)
    def calc_normals(self, pc):
        """TODO: Docstring for calc_normals.

        :pc: TODO
        :returns: TODO

        """
        # print("\tCalculating surface normals. \n")
        normals = []
        ind_of_neighbors = []
        N = len(pc)

        # tree_A = scipy.spatial.cKDTree(pc)
        # dist, nn_inds = tree_A.query(pc, k=self._nneighbors+1)
        # print('************',len(nn_inds),len(nn_inds[0]),nn_inds[0][0])
        # print('************',len(dist),len(dist[0]),dist[0][0])
        nn_inds=self._ind
        for i in range(N):
            
            # start = time.process_time()
            # Get the indices of neighbors, it is a list of tuples (dist, indx)
            indN = nn_inds[i] #<- old code
            #indN = list((neigh.kneighbors(pc[i].reshape(1, -1), return_distance=False)).flatten())
            #indN.pop(0)
            # print(i)
            
            # print(time.process_time() - start)
            # start = time.process_time()
            
            # Breakout just the indices
            # print(indN)
            # print(ind_of_neighbors)
            # indN = [indN[i][1] for i in range(len(indN))] #<- old code
            # ind_of_neighbors.append(indN[i])
            
            # print(len(indN))
            # PCA
            # print(indN)
            X = pc[indN,:].T
            # print(X.shape)
            X = X - np.mean(X, axis=1).reshape((3,1))
            # print('m:',X.shape)
            cov = np.matmul(X, X.T)/(len(indN))
            # print('cov:',cov.shape)
            _, _, Vt = np.linalg.svd(cov)
            normal = Vt[2, :]

            # Re-orient normal vectors
            if np.matmul(normal, -1.*(pc[i])) < 0:
                normal = -1.*normal
            normals.append(normal)
            
            # print(time.process_time() - start)
            # aa=5
        return normals, ind_of_neighbors
    
    
    
    def calc_normals2(self, pc):
        """TODO: Docstring for calc_normals.

        :pc: TODO
        :returns: TODO

        """
        # print("\tCalculating surface normals. \n")
        normals = []
        ind_of_neighbors = []
        ind_of_neighbors2 = []
        N = len(pc)


        import open3d as o3d 

        # pp=np.array(pc).reshape((len(pc),3))
        # source = o3d.geometry.PointCloud()
        # source.points = o3d.utility.Vector3dVector(pp)
        # pcd_tree = o3d.geometry.KDTreeFlann(source)

        pp=np.array(pc).reshape((len(pc),3))
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(pp)
        pcd_tree = o3d.geometry.KDTreeFlann(source)


        for i in range(N):
            [kk, idxx, aa] = pcd_tree.search_knn_vector_3d(source.points[i], self._nneighbors + 1)
            ind_of_neighbors.append(np.array(idxx)[1:].tolist())


#         o3d.geometry.PointCloud.estimate_normals(source,
#                 search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self._radius,
#                                                                   max_nn=self._nneighbors + 1 ))
        # o3d.geometry.estimate_normals(source,
        #         search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self._radius,
        #                                                           max_nn=self._nneighbors + 1 ))
        # o3d.cuda.pybind.t.geometry.PointCloud.estimate_normals(source,radius=self._radius,max_nn=self._nneighbors + 1 )
        
        source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self._radius, max_nn=self._nneighbors + 1))

        normals1=np.asarray(source.normals)
        # normals2=pptk.estimate_normals(pp,self._nneighbors,self._radius)


        normals1=normals1*((-2*np.where(np.diagonal(normals1.dot(pp.T))<0,0,1))+1)[:,np.newaxis]
        normals = [normals1[i,:] for i in range(len(normals1))]


        return normals, ind_of_neighbors


    #@overload_method(pptk.kdtree._build, 'clone', jit_options={'cache': True, 'parallel': True, 'nogil': True, 'boundscheck': False, 'inline': 'always'})
    #@overload_method(pptk.kdtree._build, 'clone')
    def calc_normals3(self, pp):
        """TODO: Docstring for calc_normals.

        :pc: TODO
        :returns: TODO

        """
        # print("\tCalculating surface normals. \n")
        normals = []
        ind_of_neighbors = []
        ind_of_neighbors2 = []
        N = len(pp)
        #pp=np.array(pc).reshape((len(pc),3))


        #import pptk

        # from scipy.io import savemat,loadmat
        # a=loadmat("p.mat")
        # hS=a["hS"]
        # normS=a["normS"]
        # normS=normS.reshape((len(normS),3))
        bb=pptk.kdtree._build(pp,8)
        
        for i in range(N):

            ind_of_neighbors2.append(pptk.kdtree._query(bb,i,self._nneighbors)[0].tolist())


        # o3d.geometry.PointCloud.estimate_normals(source,
                # search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self._radius*10,
                                                                  # max_nn=self._nneighbors + 1 ))
        # normals1=np.asarray(source.normals)
        normals1=pptk.estimate_normals(pp,self._nneighbors,self._radius)


        normals1=normals1*((-2*np.where(np.diagonal(normals1.dot(pp.T))<0,0,1))+1)[:,np.newaxis]
        normals = [normals1[i,:] for i in range(len(normals1))]


        return normals, ind_of_neighbors2


    def calcHistArray(self, pc, norm, indNeigh):
        """Overriding base PFH to FPFH"""
        
        # find the division thresholds for the histogram
        thresholds = self.calc_thresholds()
        
        # print("\tCalculating histograms fast method \n")
        N = len(pc)
        histArray = np.zeros((N, self._div**3))
        distArray = np.zeros((self._nneighbors))
        distList = []
        distList = typed.List(distList)
        
        for i in range(N):
            u = np.asarray(norm[i].T).squeeze()

            features = np.zeros((len(indNeigh[i]), 3))
            for j in range(len(indNeigh[i])):
                pi = pc[i]
                pj = pc[indNeigh[i][j]]
                # print(i,j)
                if np.arccos(np.dot(norm[i], (pj - pi)/(np.sqrt(np.sum(np.multiply((pj - pi),(pj - pi))))))) <= np.arccos(np.dot(norm[j],  (pi - pj)/(np.linalg.norm(pi - pj)))):
                    ps = pi
                    pt = pj
                    ns = np.asarray(norm[i]).squeeze()
                    nt = np.asarray(norm[indNeigh[i][j]]).squeeze()
                else:
                    ps = pj
                    pt = pi
                    ns = np.asarray(norm[indNeigh[i][j]]).squeeze()
                    nt = np.asarray(norm[i]).squeeze()

                u = ns
                difV = pt - ps
                dist = np.linalg.norm(difV)
                difV = difV/dist
                difV = np.asarray(difV).squeeze()
                v = np.cross(difV, u)
                w = np.cross(u, v)

                alpha = np.dot(v, nt)
                phi = np.dot(u, difV)
                theta = np.arctan(np.dot(w, nt) / np.dot(u, nt))

                features[j, 0] = alpha
                features[j, 1] = phi
                features[j, 2] = theta
                distArray[j] = dist

            distList.append(distArray)
            pfh_hist, bin_edges = self.calc_pfh_hist(features,thresholds)
            histArray[i, :] = pfh_hist / (len(indNeigh[i]))

        fast_histArray = np.zeros_like(histArray)
        for i in range(N):
            k = len(indNeigh[i])
            for j in range(k):
                spfh_sum = histArray[indNeigh[i][j]]*(1/distList[i][j])

            fast_histArray[i, :] = histArray[i, :] + (1/k)*spfh_sum
        return fast_histArray

    

# @staticmethod
# @jit(nopython=True) 
# @njit(parallel=True)
def calcHistArray2(e, div, nneighbors, rad, pc, norm, indNeigh):
    """Overriding base PFH to FPFH"""

    # find the division thresholds for the histogram
    thresholds = calc_thresholds(e, div, nneighbors, rad)

    # print("\tCalculating histograms fast method \n")
    N = len(pc)
    histArray = np.zeros((N, div**3))
    distArray = np.zeros((nneighbors))
    distList = []

    for i in range(N):
        # print('7777: ',norm[i].shape)
        u = np.asarray(norm[i].reshape(1,-1))
#         u=u.squeeze()
        # print(u.shape)

        features = np.zeros((len(indNeigh[i]), 3))
        for j in range(len(indNeigh[i])):
            pi = pc[i]
            pj = pc[indNeigh[i][j]]

            
            if np.arccos(np.dot(norm[i], (pj - pi)/(np.sqrt(np.sum(np.multiply((pj - pi),(pj - pi))))))) <= np.arccos(np.dot(norm[j],  (pi - pj)/(np.linalg.norm(pi - pj)))):
                ps = pi
                pt = pj
                ns = np.asarray(norm[i])
#                 ns = ns.squeeze()
                nt = np.asarray(norm[indNeigh[i][j]])
#                 nt = nt.squeeze()
            else:
                ps = pj
                pt = pi
                ns = np.asarray(norm[indNeigh[i][j]])
#                 ns = ns.squeeze()
                nt = np.asarray(norm[i])
#                 nt = nt.squeeze()

            u = ns
            difV = pt - ps
            dist = np.linalg.norm(difV)
#             print('dist: ',dist)
            difV = difV/dist
            difV = np.asarray(difV)
#             difV = difV.squeeze()
            v = np.cross(difV, u)
            w = np.cross(u, v)

            alpha = np.dot(v, nt)
            phi = np.dot(u, difV)
#             print('np.dot(u, nt)): ',np.dot(u, nt))
            theta = np.arctan(np.dot(w, nt) / np.dot(u, nt)) if np.dot(u, nt)!=0 else np.arctan(999999)
            
            
            features[j, 0] = alpha
            features[j, 1] = phi
            features[j, 2] = theta
            distArray[j] = dist

        distList.append(distArray)
        pfh_hist, bin_edges = calc_pfh_hist(e, div, nneighbors, rad, features,thresholds)
        histArray[i, :] = pfh_hist / (len(indNeigh[i]))
        
        
    fast_histArray = np.zeros_like(histArray)
    for i in range(N):
        k = len(indNeigh[i])
        for j in range(k):
            spfh_sum = histArray[indNeigh[i][j]]

        fast_histArray[i, :] = histArray[i, :] + (1/k)*spfh_sum
    return fast_histArray


@jit(nopython=True)
# @njit(parallel=True)
def step(e, div, nneighbors, rad, si, fi):
    """Helper function for calc_pfh_hist. Depends on selection of div

    :si: TODO
    :fi: TODO
    :returns: TODO

    """
    if div==2:
        if fi < si[0]:
            result = 0
        else:
            result = 1
    elif div==3:
        if fi < si[0]:
            result = 0
        elif fi >= si[0] and fi < si[1]:
            result = 1
        else:
            result = 2
    elif div==4:
        if fi < si[0]:
            result = 0
        elif fi >= si[0] and fi < si[1]:
            result = 1
        elif fi >= si[1] and fi < si[2]:
            result = 2
        else:
            result = 3
    elif div==5:
        if fi < si[0]:
            result = 0
        elif fi >= si[0] and fi < si[1]:
            result = 1
        elif fi >= si[1] and fi < si[2]:
            result = 2
        elif fi >= si[2] and fi < si[3]:
            result = 3
        else:
            result = 4
    return result

# @jit(nopython=True)
# @njit(parallel=True)
def calc_thresholds(e, div, nneighbors, rad, ):
    """
    :returns: 3x(div-1) array where each row is a feature's thresholds
    """
    delta = 2./div
    s1 = np.array([-1+i*delta for i in range(1,div)])
  
    delta = 2./div
    s3 = np.array([-1+i*delta for i in range(1,div)])

    delta = (np.pi)/div
    s4 = np.array([-np.pi/2 + i*delta for i in range(1,div)])

    print(s1)
    print(s3)
    print(s4)

    # s=np.concatenate((s1,s3,s4),0).reshape(3,1)
    s = np.array([s1,s3,s4])
    return s 


@jit(nopython=True)
# @njit(parallel=True)
def calc_pfh_hist(e, div, nneighbors, rad, f,s):
    """Calculate histogram and bin edges.

    :f: feature vector of f1,f3,f4 (Nx3)
    :returns:
        pfh_hist - array of length div^3, represents number of samples per bin
        bin_edges - range(0, 1, 2, ..., (div^3+1)) 
    """
    # preallocate array sizes, create bin_edges
    pfh_hist, bin_edges = np.zeros(div**3), np.arange(0,div**3+1)

    # # find the division thresholds for the histogram
    # s = self.calc_thresholds()

    # Loop for every row in f from 0 to N
    for j in range(0, f.shape[0]):
        # calculate the bin index to increment
        index = 0
        for i in range(1,4):
            index += step(e, div, nneighbors, rad, s[i-1, :], f[j, i-1]) * (div**(i-1))

        # Increment histogram at that index
        pfh_hist[index] += 1

    return pfh_hist, bin_edges



@jit(nopython=True)
# @torch.jit.script
def calc_normals(_nneighbors, _radius, pc):
    """TODO: Docstring for calc_normals.

    :pc: TODO
    :returns: TODO

    """
    # print("\tCalculating surface normals. \n")
    normals = []
    ind_of_neighbors = []
    N = len(pc)
    for i in range(N):
        
        # start = time.process_time()
        # Get the indices of neighbors, it is a list of tuples (dist, indx)
        indN = getNeighbors(_nneighbors, _radius, pc[i,:], pc) #<- old code
        #indN = list((neigh.kneighbors(pc[i].reshape(1, -1), return_distance=False)).flatten())
        #indN.pop(0)
        # print(i)
        
        # print(time.process_time() - start)
        # start = time.process_time()
        
        # Breakout just the indices
        indN = [indN[i][1] for i in range(len(indN))] #<- old code
        ind_of_neighbors.append(indN)
        
        # print(len(indN))
        # PCA
        X = pc[indN,:].T
        X = X - np.mean(X, axis=1).reshape((3,1))
        cov = np.matmul(X, X.T)/(len(indN))
        _, _, Vt = np.linalg.svd(cov)
        normal = Vt[2, :]

        # Re-orient normal vectors
        if np.matmul(normal, -1.*(pc[i])) < 0:
            normal = -1.*normal
        normals.append(normal)
        
        # print(time.process_time() - start)
        # aa=5
    return normals, ind_of_neighbors


@jit(nopython=True)
# @torch.jit.script
def getNeighbors(_nneighbors, _radius, pq, pc):
        """Get k nearest neighbors of the query point pq from pc, within the radius

        :pq: TODO
        :pc: TODO
        :returns: TODO

        """
        k = _nneighbors
        neighbors = []
        for i in range(len(pc)):
            dist = np.linalg.norm(np.subtract(pq,pc[i,:]))
            if dist <= _radius: #0.005
                neighbors.append((dist, i))
        #print("Found {} neighbors".format(len(neighbors)))
        neighbors.sort(key=lambda x:x[0])
        neighbors.pop(0)
        return neighbors[:k]
