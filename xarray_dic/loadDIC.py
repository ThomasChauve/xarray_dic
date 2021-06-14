import xarray as xr
import numpy as np
import os


def load7D(adr_data, resolution, time_step,unit_time='min',unit_res='millimeter'):
        '''
        :param adr_data: folder path where the 7D output are store
        :type adr_data: str
        :param resolution: pixel size of the image used for DIC (millimeters)
        :type resolution: float
        :param time_step: time step between the picture (seconds)
        :type time_step: float
        :param adr_micro: path for the black and white skeleton of the microstructure (bmp format) (default 0 - no microstructure)
        :type adr_micro: str
        '''
        
        # include time_step in the object
        time=time_step
                
        # find out file from 7D
        output=os.listdir(adr_data)
        output.sort()
        # loop on all the output
        all_sT=[]
        all_da=[]
        for i in list(range(len(output))):
            # load the file
            data=np.loadtxt(adr_data + '/' + output[i])
            # find not indexed point
            id=np.where(data[:,4]!=0.0)
            # replace not indexed point by NaN value
            data[id,2:4]=np.NaN
            data[id,5:8]=np.NaN
            # for the first step it extract size sx,sy and window correlation value used in 7D
            if (i==0):
                n_dic=np.abs(data[0,1]-data[1,1])
                nb_pix=np.size(data[:,0])
                sx=np.int32(np.abs(data[0,0]-data[nb_pix-1,0])/n_dic+1)
                sy=np.int32(np.abs(data[0,1]-data[nb_pix-1,1])/n_dic+1)
                eij=np.zeros([sy,sx])
                eij[:,:]=np.nan
                
            # Build image at time t=i
            u=np.transpose(np.reshape(data[:,2]*resolution,[sx,sy]))
            v=np.transpose(np.reshape(data[:,3]*resolution,[sx,sy]))
            exx=np.transpose(np.reshape(data[:,5],[sx,sy]))
            eyy=np.transpose(np.reshape(data[:,6],[sx,sy]))
            exy=np.transpose(np.reshape(data[:,7],[sx,sy]))
            all_sT.append(np.dstack((exx,eyy,eij,exy,eij,eij)))
            all_da.append(np.dstack((u,v)))
            dx=np.linspace(0,n_dic*resolution*(sx-1),sx)
            dy=np.linspace(0,n_dic*resolution*(sy-1),sy)
            
        dsT=xr.DataArray(np.stack(all_sT),dims=['time','y','x','sT'])
        da=xr.DataArray(np.stack(all_da),dims=['time','y','x','d'])

        time=np.linspace(time_step,time_step*len(output),len(output))

        ds = xr.Dataset(
        {   
            "displacement": da,
            "strain": dsT,

        },
        )
        ds.coords['x']=dx
        ds.coords['y']=dy
        ds.coords['time']=time

        ds.attrs["unit_time"]=unit_time
        ds.attrs["step_size"]=resolution
        ds.attrs["unit_position"]=unit_res
        ds.attrs["window_size"]=int(n_dic)
        ds.attrs["path_dat"]=adr_data
        ds.attrs["DIC_software"]='7D'
        
        return ds
    

def loadDICe(adr_data, resolution, time_step,unit_time='min',unit_res='millimeter',strain_com=1):
    '''
    :param adr_data: folder path where the DICe output are store
    :type adr_data: str
    :param resolution: pixel size of the image used for DIC (millimeters)
    :type resolution: float
    :param time_step: time step between the picture (seconds)
    :type time_step: float
    :param adr_micro: path for the black and white skeleton of the microstructure (bmp format) (default 0 - no microstructure)
    :type adr_micro: str
    :param strain_com: Computation method for strain calculation 0-small deformation 1-Green Lagrange (default 1)
    :type strain_com: bool
    '''

    # include time_step in the object
    time=time_step

    # find out file from 7D
    output=os.listdir(adr_data+'results/')
    output.sort()
    output=output[2:-1]
    # loop on all the output
    all_sT=[]
    all_da=[]
    for i in list(range(len(output))):
    #for i in [0]:
        # load the file
        data=np.loadtxt(adr_data+'results/'+ output[i],skiprows=1,delimiter=',',usecols=[1,2,3,4,9])
        # find not indexed point
        id=np.where(data[:,-1]==0)
        # replace not indexed point by NaN value
        data[id[0],2]=np.NaN
        data[id[0],3]=np.NaN
        # for the first step it extract size sx,sy and window correlation value used in 7D
        if (i==0):
            n_dic=np.abs(data[0,1]-data[1,1])
            nb_pix=np.size(data[:,0])
            sx=np.int32(1+np.abs(np.min(data[:,0])-np.max(data[:,0]))/n_dic)
            sy=np.int32(1+np.abs(np.min(data[:,1])-np.max(data[:,1]))/n_dic)
            dx=np.linspace(0,n_dic*resolution*(sx-1),sx)
            dy=np.linspace(0,n_dic*resolution*(sy-1),sy)
        
        # Build image at time t=i
        tmp_u=np.zeros([sx,sy])
        tmp_u[:,:]=np.NaN
        tmp_v=np.zeros([sx,sy])
        tmp_v[:,:]=np.NaN
        # find x y position in the table
        x=(data[:,0]-np.min(data[:,0]))/n_dic
        y=(data[:,1]-np.min(data[:,1]))/n_dic
        # include data
        tmp_u[x.astype(int),y.astype(int)]=data[:,2]
        tmp_v[x.astype(int),y.astype(int)]=data[:,3]
        # Create im2d data object u and v
        imu=np.transpose(tmp_u*resolution)
        imv=np.transpose(tmp_v*resolution)
                
        da=xr.DataArray(np.dstack((imu,imv)),dims=['y','x','vc'])
        da.coords['x']=dx
        da.coords['y']=dy
        all_da.append(np.dstack((imu,imv)))
        
        diff_ux=da[:,:,0].diff('x')/(resolution*n_dic)
        diff_ux=diff_ux.where(diff_ux.y>0,drop=True)
        diff_vx=da[:,:,1].diff('x')/(resolution*n_dic)
        diff_vx=diff_vx.where(diff_vx.y>0,drop=True)
        
        diff_uy=da[:,:,0].diff('y')/(resolution*n_dic)
        diff_uy=diff_uy.where(diff_uy.x>0,drop=True)
        diff_vy=da[:,:,1].diff('y')/(resolution*n_dic)
        diff_vy=diff_vy.where(diff_vy.x>0,drop=True)
        
        empty=np.zeros(diff_ux.shape[0:2])
        empty[:,:]=np.nan
        # exx, eyy, ezz, exy, exz, eyz
        if strain_com==0:
            sT=np.dstack((
                np.array(diff_ux),
                np.array(diff_vy),
                empty,
                np.array((diff_uy+diff_vx)*0.5),
                empty,
                empty,
            ))
        else:
            sT=np.dstack((
                np.array(diff_ux+(diff_ux**2+diff_vx**2)*0.5),
                np.array(diff_vy+(diff_uy**2+diff_vy**2)*0.5),
                empty,
                np.array((diff_uy+diff_vx)*0.5+(diff_ux*diff_uy+diff_vx*diff_vy)*0.5),
                empty,
                empty,
            ))
        
        all_sT.append(sT)
       
    dsT=xr.DataArray(np.stack(all_sT),dims=['time','yt','xt','sT'])
    da=xr.DataArray(np.stack(all_da),dims=['time','y','x','d'])
    
    time=np.linspace(time_step,time_step*len(output),len(output))
    
    ds = xr.Dataset(
    {   
        "displacement": da,
        "strain": dsT,

    },
    )
    ds.coords['x']=dx
    ds.coords['y']=dy
    ds.coords['xt']=dx[1::]
    ds.coords['yt']=dy[1::]
    ds.coords['time']=time

    ds.attrs["unit_time"]=unit_time
    ds.attrs["step_size"]=resolution
    ds.attrs["unit_position"]=unit_res
    ds.attrs["window_size"]=int(n_dic)
    ds.attrs["path_dat"]=adr_data
    ds.attrs["DIC_software"]='DICe'
            
            
    return ds#,sT,oxy