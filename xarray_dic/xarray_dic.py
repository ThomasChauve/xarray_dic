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
#--------------------geometric transformation---------------------------

    def DIC_line(self,axis='x',shift=3):
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