#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

'''
Python function for reading post-processing files produced
by caffa3d.MBRi and caffa3d.H at different region levels and multi-grid levels.
Originally coded in MATLAB by Gabriel Usera.

version 21_0009: 

    - Using ypstruct package - FL and GR are lists of yp-structus
    - FL and GR are accesed with (x, y, z)
    - Incorporates Multi-Block, structs are accesed as FL[reg_no][blk_no].Field
    
Pending:

    - Multi-grid levels, block-counter
    - MergeBlocks Flag:
        - include a flag to merge blocks belonging to the same region
        - Direction of blocks must be identified first
'''

__author__ = "Paolo Sassi"
__email__  = "psassi@fing.edu.uy"
__status__ = "Development"
__date__   = "09/21"

#
import os
import shutil
import glob
import numpy as np
import gzip
from shutil import copy2
from ypstruct import struct
# from src.post.caffa2VTK_21_0003 import caffa2VTK
from IPython.display import clear_output
#
# ----------------------------------------------------------------------------- #
# Function for reading post-processing files
# ----------------------------------------------------------------------------- #
def caffa3dMBRi_gz(
    Folder,
    FileNameRoot,
    Regions,
    MgLevels,
    nFiles,
    GridFlag=True,
    FieldList=['Xc','Vol','Fm','U','UMean','UUMean','P','T','Ximb','Dimb','Vis','Tr','Vof'],
    VTK = False,
    decimals=5,
    ):

    """
    Function for reading post-processing files produced
    by caffa3d.MBRi at different region levels and multi-grid levels.
    
     Parameters
    -----------

    Folder: string
        Paths to post-processing files *.gz.

    FileNameRoot: string
        Case name.

    Regions: [n] list of integers
        Regions to read. If empty all regions will be read.

    MgLevels = [n] list of integers
        Multi grid level to read.
        
    nFiles = [n] list of integers
        File number to read, correspond to the number of time steps.
        
    GridFlag: boolean
        Indicates if grid file will be read. Default value is "True".
    
    FieldList: [n] list of strings
        List of all the fields to be extracted from the post-processing files

    VTK: boolean
        Indicates if VTK files should be created. Default value is "False".
    
    decimals: integer
        If VTK is True, 
        Number of decimals to round the cells positions (Xc) in vtk files.
        Default value is 5
        
        
    Returns
    -------

    FL: [nRegions] list of yp-structs
        Data structure with fields
    
    GR: [nRegions] list of yp-structs
        Data structure with grid data.

    Tips
    ----

    If there are one Block per Region FL and GR are accesed as:

        FL[reg_no].Field

    If Multi-Block is activated (several blocks per region):

        FL[reg_no][blk_no].Field
    """

    # Possible errors
    # ------------------------------------------------------------------------- #

    # nFiles is a list?
    if type(nFiles) != list:
        nFiles = [int(nFiles)]
        # print("nFiles type was changed to list")

    # Folder must end with a slash
    if not os.name == 'nt':
        if Folder[-1] != '/':
            Folder = Folder + '/'

    # Does Folder exists?
    if not os.path.isdir(Folder):
        raise NameError('Given Folder does not exist, please check!')

    # Are there *gz files in Folder?
    outputs_path = os.path.join(Folder, "*gz")
    files = glob.glob(outputs_path)
    files.sort()
    nItems = len(files)
    if nItems == 0:
        raise NameError('No *gz files in {}'.format(Folder))
    # ------------------------------------------------------------------------- #
    #
    # Start function caffa3dMBRi_gz
    #
    StandardFileNameLength=[33,    # test01.rgc.001.mgc.001.flc.020.gz
                            35,    # test02.rgc.001.mgc.001.flc.00020.gz
                            37];   # test03.rgc.001.mgc.001.flc.0000020.gz
    #
    FilesListed=[]
    nRegions = 0
    #
    for jItem in range(nItems):
        if not os.name == 'nt':
            ItemName=files[jItem][len(Folder):]
        else:
            ItemName=files[jItem][len(Folder)+1:]
        nItemName=len(ItemName)
        nRegions = np.max([nRegions,int(ItemName[11:14])])
        if nItemName in StandardFileNameLength:
            if FileNameRoot in ItemName:
                FilesListed.append(ItemName)
    #
    FL = [None] * nRegions
    GR = [None] * nRegions
    #
    if not FilesListed:
        raise NameError("No matching files found")
    #
    for jListedFile in range(len(FilesListed)):
        jListedFileName=FilesListed[jListedFile]
        jFileRegionCounter = int(jListedFileName[11:14])
        jFileMgLevelCounter= int(jListedFileName[19:22])
        jFileOutputCounter = int(jListedFileName[27:-3])
        FileWasRequested=1
        #
        if Regions:
            if jFileRegionCounter not in Regions:
                FileWasRequested=0
            #
        if MgLevels:
            if jFileMgLevelCounter not in MgLevels:
                FileWasRequested=0
            #
        ThisIsGridFile=0
        if GridFlag:
            if 'grd' in jListedFileName:
                ThisIsGridFile=1
            #
        if nFiles:                                 # File numbers were specified ?
            if jFileOutputCounter not in nFiles:   # 'this' file number was requested
                if jFileOutputCounter>0:           # no, is file number > 0
                    FileWasRequested=0             # yes, discard this file
                else:                              # no, file number == 0
                    if not ThisIsGridFile:
                        FileWasRequested=0
        #
        if FileWasRequested:
            FL, GR = LoadThisFile(
                FL,
                GR,
                Folder,
                jListedFileName,
                jFileRegionCounter,
                jFileMgLevelCounter,
                jFileOutputCounter,
                ThisIsGridFile,
                FieldList
            )
        #
    # clear_output(wait=False)
    # print("FL and GR ready!")
    if VTK:
        caffa2VTK(Folder,FileNameRoot,FL,GR,FieldList,nFiles[0],decimals=decimals)

        # print("Done with VTK files!")
    #
    return FL, GR
#
# ----------------------------------------------------------------------------- #
# Function for loading file
# ----------------------------------------------------------------------------- #
def LoadThisFile(
    FL,
    GR,
    Folder,
    FileName,
    Region,
    MgLevel,
    OutputNumber,
    ThisIsGridFile,
    FieldList):
    #
    # Open file. Linux
    FilePath = os.path.join(Folder,FileName)
    copy2(FilePath, './')
    # Decompress the file
    with gzip.open(FileName, 'rb') as f_in:
        with open(FileName[:-3], 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    # Remove the copied compressed file
    os.remove(FileName)

    # print('Opened file: ',FileName)
    
    # Read Real Type parameters to detect single/double precision
    off = 4
    IrlKind = np.fromfile(FileName[:-3],dtype=np.int32,count=1,offset=off)[0]
    off +=4
    if IrlKind == 0:
        dumword = np.int64
        dum_off = 8
        IrlKind= np.fromfile(FileName[:-3],dtype=np.int32,count=1,offset=off)[0]
        off += 4
    else:
        dumword = np.int32
        dum_off = 4
    #
    IrlKind2 = np.fromfile(FileName[:-3],dtype=np.int32,count=1,offset=off)[0]
    off += 4
    off += dum_off

    if IrlKind==IrlKind2:
        VarWord= np.float64
        var_off = 8
        # print('Double precision selected')
    else:
        VarWord = np.float32
        var_off = 4
        # print('Single precision selected')
    #
    # Read dimensions and parameters
    off += dum_off
    Itim    = np.fromfile(FileName[:-3],dtype=np.int32,count=1,offset=off)[0]
    off += 4
    Time    = np.fromfile(FileName[:-3],dtype=VarWord,count=1,offset=off)[0]
    off += var_off
    dt      = np.fromfile(FileName[:-3],dtype=VarWord,count=1,offset=off)[0]
    off += var_off
    NblksRG = np.fromfile(FileName[:-3],dtype=np.int32,count=1,offset=off)[0]
    off += 4
    MaxIJK  = np.fromfile(FileName[:-3],dtype=np.int32,count=1,offset=off)[0]
    off += 4
    NFI     = np.fromfile(FileName[:-3],dtype=np.int32,count=1,offset=off)[0]
    off += 4
    off += dum_off
    #
    off += dum_off
    NiBK    = np.fromfile(FileName[:-3],dtype=np.int32,count=NblksRG,offset=off)
    off += 4*NblksRG
    NjBK    = np.fromfile(FileName[:-3],dtype=np.int32,count=NblksRG,offset=off)
    off += 4*NblksRG
    NkBK    = np.fromfile(FileName[:-3],dtype=np.int32,count=NblksRG,offset=off)
    off += 4*NblksRG
    off += dum_off
    #
    off += dum_off
    iBK     = np.fromfile(FileName[:-3],dtype=np.int32,count=NblksRG,offset=off)
    off += 4*NblksRG
    jBK     = np.fromfile(FileName[:-3],dtype=np.int32,count=NblksRG,offset=off)
    off += 4*NblksRG
    kBK     = np.fromfile(FileName[:-3],dtype=np.int32,count=NblksRG,offset=off)
    off += 4*NblksRG
    off += dum_off
    #
    off += dum_off
    NijkBRG = np.fromfile(FileName[:-3],dtype=np.int32,count=NblksRG,offset=off)
    off += 4*NblksRG
    ijkBRG  = np.fromfile(FileName[:-3],dtype=np.int32,count=NblksRG,offset=off)
    off += 4*NblksRG
    off += dum_off
    #
    off += dum_off
    kMgLevel  = np.fromfile(FileName[:-3],dtype=np.int32,count=1,offset=off)[0]
    off += 4
    NmgLevels = np.fromfile(FileName[:-3],dtype=np.int32,count=1,offset=off)[0]
    off += 4
    NblksMG   = np.fromfile(FileName[:-3],dtype=np.int32,count=1,offset=off)[0]
    off += 4
    iBlksMGRG = np.fromfile(FileName[:-3],dtype=np.int32,count=NmgLevels,offset=off)
    off += 4*NmgLevels
    iStrMGRG  = np.fromfile(FileName[:-3],dtype=np.int32,count=NmgLevels,offset=off)
    off += 4*NmgLevels
    iEndMGRG  = np.fromfile(FileName[:-3],dtype=np.int32,count=NmgLevels,offset=off)
    off += 4*NmgLevels
    off += dum_off
    #
    off += dum_off
    LwGrid  = np.fromfile(FileName[:-3],dtype=np.int32,count=1,offset=off)[0]
    off += 4
    LwGridXr= np.fromfile(FileName[:-3],dtype=np.int32,count=1,offset=off)[0]
    off += 4
    LwGridBc= np.fromfile(FileName[:-3],dtype=np.int32,count=1,offset=off)[0]
    off += 4
    LwFm    = np.fromfile(FileName[:-3],dtype=np.int32,count=1,offset=off)[0]
    off += 4
    LwGrad  = np.fromfile(FileName[:-3],dtype=np.int32,count=1,offset=off)[0]
    off += 4
    LwResmon= np.fromfile(FileName[:-3],dtype=np.int32,count=1,offset=off)[0]
    off += 4
    off += dum_off
    #
    off += dum_off
    LwCal = np.fromfile(FileName[:-3],dtype=np.int32,count=NFI,offset=off)
    off += 4*NFI
    off += dum_off
    #
    # Read Basic Grid Data and sort in blocks
    if LwGrid:
        X, off, j = ReadFieldFromFile(
            FileName[:-3],'single',off,4,dum_off,'X'  ,3,Region,MgLevel,
            iBlksMGRG,NblksMG,NijkBRG,ijkBRG,NkBK,NiBK,NjBK,
            iStrMGRG,iEndMGRG,FieldList
            )
        #
        off -= dum_off*2
        Xc, off, j = ReadFieldFromFile(
            FileName[:-3],'single',off,4,dum_off,'Xc' ,3,Region,MgLevel,
            iBlksMGRG,NblksMG,NijkBRG,ijkBRG,NkBK,NiBK,NjBK,
            iStrMGRG,iEndMGRG,FieldList
            )
        #
        off -= dum_off*2
        Vol, off, j = ReadFieldFromFile(
            FileName[:-3],'single',off,4,dum_off,'Vol',1,Region,MgLevel,
            iBlksMGRG,NblksMG,NijkBRG,ijkBRG,NkBK,NiBK,NjBK,
            iStrMGRG,iEndMGRG,FieldList
            )
        #
        GridSave = [field for field in FieldList if field in ['X','Xc','Vol']]
        
        # Add struct to the list GR, 
        # considering if there are several blocks in current region
        if NblksMG == 1:
            # Only one block in current region
            dic_aux = {}
            for i in range(len(GridSave)):
                dic_aux[GridSave[i]] = np.squeeze(locals()[GridSave[i]])
            #
            GR[Region-1] = struct(dic_aux)
        else:
            # Several blocks in current region
            for k in range(NblksMG):
                dic_aux = {}
                for i in range(len(GridSave)):
                    dic_aux[GridSave[i]] = np.squeeze(np.split(np.array(locals()[GridSave[i]]),NblksMG,axis=0)[k])
                #
                if GR[Region-1]:
                    GR[Region-1].append(struct(dic_aux))
                else:
                    GR[Region-1] = [struct(dic_aux)]
        #
        if LwGridXr:
            off += dum_off
            #...
            off += dum_off

    # print('Done with grid data : ',FileName)

    if ThisIsGridFile:
 
        os.remove(FileName[:-3])
        return FL, GR
    
    #
    # Read Flow data and sort in blocks
    # ------------------------------------------------------------------------- #
    # Read Fm Field
    if LwFm:
        Fm, off, j = ReadFieldFromFile(
                FileName[:-3],VarWord,off,var_off,dum_off,'Fm',3,Region,MgLevel,
                iBlksMGRG,NblksMG,NijkBRG,ijkBRG,NkBK,NiBK,NjBK,
                iStrMGRG,iEndMGRG,FieldList
            )
    # Read Velocity Field
    if LwCal[0]:
        U, off, j = ReadFieldFromFile(
                FileName[:-3],VarWord,off,var_off,dum_off,'U',3,Region,MgLevel,
                iBlksMGRG,NblksMG,NijkBRG,ijkBRG,NkBK,NiBK,NjBK,
                iStrMGRG,iEndMGRG,FieldList
            )
    # Read Pressure Field
    if LwCal[3]:
        P, off, j = ReadFieldFromFile(
            FileName[:-3],VarWord,off,var_off,dum_off,'P',1,Region,MgLevel,
            iBlksMGRG,NblksMG,NijkBRG,ijkBRG,NkBK,NiBK,NjBK,
            iStrMGRG,iEndMGRG,FieldList
            )
    #
    # Read Temperature Field
    if LwCal[4]:
        T, off, j = ReadFieldFromFile(
            FileName[:-3],VarWord,off,var_off,dum_off,'T',1,Region,MgLevel,
            iBlksMGRG,NblksMG,NijkBRG,ijkBRG,NkBK,NiBK,NjBK,
            iStrMGRG,iEndMGRG,FieldList
            )
    # Read Viscosity Field
    Ivof, Ibgh = 9, 14
    if (LwCal[5] or LwCal[Ivof] or LwCal[Ibgh]):
        Vis, off, j = ReadFieldFromFile(
            FileName[:-3],VarWord,off,var_off,dum_off,'Vis',1,Region,MgLevel,
            iBlksMGRG,NblksMG,NijkBRG,ijkBRG,NkBK,NiBK,NjBK,
            iStrMGRG,iEndMGRG,FieldList
            )
    # Read Immersed Boundary Method Fields
    Imbc = 16
    if LwCal[Imbc]:
        Ximb, off, j = ReadFieldFromFile(
            FileName[:-3],VarWord,off,var_off,dum_off,'Ximb',1,Region,MgLevel,
            iBlksMGRG,NblksMG,NijkBRG,ijkBRG,NkBK,NiBK,NjBK,
            iStrMGRG,iEndMGRG,FieldList
            )
        Dimb, off, j = ReadFieldFromFile(
            FileName[:-3],VarWord,off,var_off,dum_off,'Dimb',1,Region,MgLevel,
            iBlksMGRG,NblksMG,NijkBRG,ijkBRG,NkBK,NiBK,NjBK,
            iStrMGRG,iEndMGRG,FieldList
            )
    # Read Volume of Fluid Method Field
    if LwCal[Ivof]:
        Vof, off, j = ReadFieldFromFile(
            FileName[:-3],VarWord,off,var_off,dum_off,'Vof',1,Region,MgLevel,
            iBlksMGRG,NblksMG,NijkBRG,ijkBRG,NkBK,NiBK,NjBK,
            iStrMGRG,iEndMGRG,FieldList
            )
    # Read Qv Field - Vapour fraction in Wet-Air Module
    Iqv, Iqc = 10, 11
    if LwCal[Iqv]:
        Qv, off, j = ReadFieldFromFile(
            FileName[:-3],VarWord,off,var_off,dum_off,'Qv' ,1,Region,MgLevel,
            iBlksMGRG,NblksMG,NijkBRG,ijkBRG,NkBK,NiBK,NjBK,
            iStrMGRG,iEndMGRG,FieldList
            )
    # Read Qc Field - Condensated fraction in Wet-Air Module
    if LwCal[Iqc]:
        Qc, off, j = ReadFieldFromFile(
            FileName[:-3],VarWord,off,var_off,dum_off,'Qc' ,1,Region,MgLevel,
            iBlksMGRG,NblksMG,NijkBRG,ijkBRG,NkBK,NiBK,NjBK,
            iStrMGRG,iEndMGRG,FieldList
            )
    # Read RadSrco Field
    if LwCal[Iqc]:
        RadSrco, off, j = ReadFieldFromFile(
            FileName[:-3],VarWord,off,var_off,dum_off,'RadSrco' ,1,Region,MgLevel,
            iBlksMGRG,NblksMG,NijkBRG,ijkBRG,NkBK,NiBK,NjBK,
            iStrMGRG,iEndMGRG,FieldList
            )
    # Read Tracers Field
    Itr = 17
    if LwCal[Itr]:
        Tr, off, j = ReadFieldFromFile(
            FileName[:-3],VarWord,off,var_off,dum_off,'Tr' ,1,Region,MgLevel,
            iBlksMGRG,NblksMG,NijkBRG,ijkBRG,NkBK,NiBK,NjBK,
            iStrMGRG,iEndMGRG,FieldList
            )   
    # Read Resmon Field
    if LwResmon:
        Resmon, off, j = ReadFieldFromFile(
            FileName[:-3],VarWord,off,var_off,dum_off,'Resmon',1,Region,MgLevel,
            iBlksMGRG,NblksMG,NijkBRG,ijkBRG,NkBK,NiBK,NjBK,
            iStrMGRG,iEndMGRG,FieldList
            )
    # Read Mean Velocity Field
    if LwCal[1]:
        CsgsC, off, j = ReadFieldFromFile(
            FileName[:-3],VarWord,off,var_off,dum_off,'CsgsC',1,Region,MgLevel,
            iBlksMGRG,NblksMG,NijkBRG,ijkBRG,NkBK,NiBK,NjBK,
            iStrMGRG,iEndMGRG,FieldList
            )
        betaCoef, off, j = ReadFieldFromFile(
            FileName[:-3],VarWord,off,var_off,dum_off,'betaCoef',1,Region,MgLevel,
            iBlksMGRG,NblksMG,NijkBRG,ijkBRG,NkBK,NiBK,NjBK,
            iStrMGRG,iEndMGRG,FieldList
            )
        UMean, off, j = ReadFieldFromFile(
                FileName[:-3],VarWord,off,var_off,dum_off,'UMean',3,Region,MgLevel,
                iBlksMGRG,NblksMG,NijkBRG,ijkBRG,NkBK,NiBK,NjBK,
                iStrMGRG,iEndMGRG,FieldList
            )
    # Read Reynolds Stresses Field
        UUMean, off, j = ReadFieldFromFile(
                FileName[:-3],VarWord,off,var_off,dum_off,'UUMean',6,Region,MgLevel,
                iBlksMGRG,NblksMG,NijkBRG,ijkBRG,NkBK,NiBK,NjBK,
                iStrMGRG,iEndMGRG,FieldList
            )
    #
    FieldSave = [
        field for field in FieldList if field in [
            'Fm','U','P','T','Ximb','Dimb','Vis','Tr','Vof','Resmon','UMean','UUMean','RadSrco','Qc','Qv'
            ]]
    
    # Add struct to the list FL, 
    # considering if there are several blocks in current region
    if NblksMG == 1:
        # Only one block in current region
        dic_aux = {}
        for i in range(len(FieldSave)):
            dic_aux[FieldSave[i]] = np.squeeze(locals()[FieldSave[i]])
        #
        dic_aux.update({'Time': Time, 'dt': dt})
        #
        FL[Region-1] = struct(dic_aux)
    else:
        # Several blocks in current region
        for k in range(NblksMG):
            dic_aux = {}
            for i in range(len(FieldSave)):
                dic_aux[FieldSave[i]] = np.squeeze(np.split(np.array(locals()[FieldSave[i]]),NblksMG,axis=0)[k])
            #
            dic_aux.update({'Time': Time, 'dt': dt})
            #
            if FL[Region-1]:
                FL[Region-1].append(struct(dic_aux))
            else:
                FL[Region-1] = [struct(dic_aux)]
    #
    os.remove(FileName[:-3])
    clear_output(wait=False)
    return FL, GR
    #
# ----------------------------------------------------------------------------- #
# Read Field From File Function
# ----------------------------------------------------------------------------- #
def ReadFieldFromFile(
    FileHandle,
    TypeWord,
    global_offset,
    variable_offset,
    dummy_offset,
    FieldName,
    FieldDim,
    Region,
    MgLevel,
    iBlksMGRG,
    NblksMG,
    NijkBRG,
    ijkBRG,
    NkBK,
    NiBK,
    NjBK,
    iStrMGRG,
    iEndMGRG,
    FieldList):
    
    """
    Function for extracting the specific field from the post-processing file
    by caffa3d.MBRi at different region levels and multi-grid levels.
    
     Parameters
    -----------

    FileHandle: string.
        Paths to post-processing file *.gz.

    TypeWord: string.
        Indicates single or double precision.

    global_offset: integer.
        Offset of binary file where required field is located.

    variable_offset: integer.
        Number of bits for each entry of the field in binary file.

    dummy_offset: integer.
        Number of padding bits between fields.

    Data: dataclass
        Data class to add the extracted field.

    FieldName = string.
        Name of the field to extract.
        
    FieldDim = integer
        Dimensions of current field.
        
    Region: boolean
        Region to read.
    
    MgLevel: [n] list of strings
        Multi grid level to read.
        
    iBlksMGRG: integer.

    NblksMG: integer.
        Number of blocks.

    NijkBRG: integer.
        Total number of cells.

    ijkBRG:

    NkBK: integer
        Number of cells in k-direction, current block.

    NiBK: integer
        Number of cells in i-direction, current block.

    NjBK: integer
        Number of cells in j-direction, current block.

    iStrMGRG: [] list of integers.
        First cell of currents Multi-grid and region.

    iEndMGRG: [] list of integers.
        Last cell of currents Multi-grid and region.

    FieldList: [] list of strings.
        List of all the fields to be extracted from the post-processing files.

        
    Returns
    -------

    Field: np.array [float] 
        Extracted field.

    BlockIndex

    """
    #
    Off=-iStrMGRG[MgLevel - 1]+1
    FieldCount=iEndMGRG[MgLevel - 1]-iStrMGRG[MgLevel -1 ]+1
    #
    global_offset += dummy_offset
    Data = np.fromfile(
        FileHandle,dtype=TypeWord,count=FieldDim*FieldCount,offset=global_offset
        )
    # Check Field length
    if (len(Data) != (FieldDim*FieldCount)) and (FieldName in FieldList):
        os.remove(FileHandle)
        raise ValueError(
            '''Field {} has length {} 
            check FieldList or LwCal'''.format(FieldName,len(Data))
            )
    global_offset += variable_offset * FieldDim * FieldCount
    global_offset += dummy_offset
    #
    Field = []
    BlockIndex = []
    if FieldName in FieldList:
        Data = np.reshape(Data, (FieldDim,FieldCount), order="F")
        for jBlock in range( iBlksMGRG[MgLevel-1], iBlksMGRG[MgLevel-1]+NblksMG ):
            #
            # Several blocks are treated as separate lists (append).
            Field.append(
                np.transpose(
                    np.reshape(
                        Data[:,Off+ijkBRG[jBlock]:Off+ijkBRG[jBlock]+NijkBRG[jBlock]],
                        (FieldDim,NjBK[jBlock],NiBK[jBlock],NkBK[jBlock]), order='F'
                        ),
                    (2,1,3,0)
                    )
                )
            BlockIndex.append(jBlock)
            #
            # MergeBlocks Flag, include if, determine axis for np.concatenate
    #     #
    # #
    return Field, global_offset, BlockIndex
# ------------------------------------------------------------------------- #
