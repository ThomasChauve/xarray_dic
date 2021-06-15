import xarray as xr
import numpy as np

@xr.register_dataset_accessor("dic")

class xarray_dic(object):
    '''
    This is a classe to work on aita data in xarray environnement.
    
    .. note:: xarray does not support heritage from xr.DataArray may be the day it support it, we could move to it
    '''
    
    def __init__(self, xarray_obj):
        '''
        Constructor for aita. 
        
        The xarray_obj should contained at least :
        1. displacement : DataArray that contained displacement in the last column of dimention
        2. strain : DataArray taht contained strain field compatible with xarray_symTensor2d
        
        :param xarray_obj:
        :type xarray_obj: xr.Dataset
        '''
        self._obj = xarray_obj 
    pass
#----------------------------------------------------------------------------------------------------------

    def DIC_line(self,axis='y',shift=3):
        '''
        Compute the average of one componant
        
        :param axis: txx,tyy,tzz,txy,txz,tyz
        :type axis: str
        :return mean: average of this component
        :rtype mean: xr.DataArray
        '''
        
        if axis=='x':
            sd=(np.nanmean(self._obj.displacement[...,:,-shift,0],axis=-1)-np.nanmean(self._obj.displacement[...,:,shift,0],axis=-1))/np.array((self._obj.x[-shift]-self._obj.x[shift]))
        if axis=='y':
            sd=(np.nanmean(self._obj.displacement[...,-shift,:,1],axis=-1)-np.nanmean(self._obj.displacement[...,shift,:,1],axis=-1))/np.array((self._obj.y[-shift]-self._obj.y[shift]))
            
        return xr.DataArray(sd,dims=self._obj.displacement.coords.dims[0])

#-----------------------------------------------------------------------------------------------------------
    def find_pic(self,strain_step=-0.005,a_im=1,b_im=0,**kwargs):
        '''
        Find picture number as between 2 picture the increment of deformation is macro_strain
        
        :param strain_step: macroscopic strain between 2 picture.
        :type strain_step: float
        :param a_im: value to give true output file number image a_im*n+b_im
        :type a_im: int
        :param b_im: value to give true output file number image a_im*n+b_im
        :type b_im: im
        :return: nb_img
        :rtype: array
        '''
        
        ds=self._obj
        # extract macroscopic strain from the data
        ds['dl']=self.DIC_line(**kwargs)
        
        nb_img=[]
        
        macro_line=np.array(ds.dl)
        
        id=np.isnan(macro_line)
        macro_line[id]=0

        for i in range(int(macro_line[-1]/strain_step)):
            # find where the macro strain is higher that the step
            idx=np.where(macro_line>np.float(i+1)*strain_step)
            # take the minimum
            label=np.where(macro_line==np.min(macro_line[idx]))
            nb_img.append(list(label[0]))
            
            
        res=a_im*np.array(nb_img)+b_im 
        #np.savetxt('wanted_image.txt',res,fmt='%i')
            
        return res