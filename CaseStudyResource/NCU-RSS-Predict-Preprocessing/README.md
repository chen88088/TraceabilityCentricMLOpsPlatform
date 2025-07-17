# NCU-RSS-Predict-Preprocessing

# Usage: RSS 1.3 
Converts aerial image .tif files into .png, then crops the .png into framelets. Image enhancement similar to the one in PNGProducing is applied. 
## input
- A single TIF file should be associated with the following:
```
94191004_181006z_14~4907_hr4.dxf
94191004_181006z_14~4907_hr4.tfw
94191004_181006z_14~4907_hr4.tif
```
- Place the .tif of interest, together with its associated files under `./TIF`.  

## Run
- py prepare_inference_data_13.py
## Results
- `./data/Predict_IMG_CROP` will contain the results. It should contain 2160*N framelets.

# Usage RSS 1.5
- This use case runs on `arcpy`, so it requires a Windows machine!
- Converts aerial image .tif files into .png, and shp into parcel mask pngs.
## input
### TIF file.  
- A single TIF file should be associated with the following:
```
94191004_181006z_14~4907_hr4.dxf
94191004_181006z_14~4907_hr4.tfw
94191004_181006z_14~4907_hr4.tif
```
- Place the .tif of interest, together with its associated files under `./TIF`.  
### SHP file
- An aerial frame is paired with a shp file. A shp file is asscociated with the following:
```ps
PD_94191004_181006z_14~4907_hr4.cpg
PD_94191004_181006z_14~4907_hr4.dbf
PD_94191004_181006z_14~4907_hr4.prj
PD_94191004_181006z_14~4907_hr4.sbn
PD_94191004_181006z_14~4907_hr4.sbx
PD_94191004_181006z_14~4907_hr4.shp
PD_94191004_181006z_14~4907_hr4.shp.xml
PD_94191004_181006z_14~4907_hr4.shx
```
- The shp file needs to have a column named `Label_Num`. This is used to locate parcels in the mask. If the shp file currently does not have a column named `Label_Num`, go to `NCU-RSS-SHP-Writer` and run `generate_Label_Num_and_set_random_values` to generate this column.

- Place the .shp having the column named `Label_Num`, together with its associated files under `./SHP`.  

## Run
```powershell
C:\'Program Files'\ArcGIS\Pro\bin\Python\envs\arcgispro-py3\python.exe ./prepare_inference_data_15.py
```
## Results
- `data\predict_NIRRG` and `data\RSS15_Training_rice_mask` will contain the results. Each will contain N .png files, Where N is the number of aerial image frames to predict in this batch.