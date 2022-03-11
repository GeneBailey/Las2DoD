# Las2Dod #

The goal of this tool is to take 2 LAS point clouds and automatically compute a Dem of Difference while incorporating uncertainty testing through a cell-by-cell Welch's t-test. 

Currently the tool is implemented through 
* a python command line script
* a Standalone GUI Executable

## Standalone GUI ##
A standalone executable was built from the source code @main_gui.py using tkinter and pyinstaller. 

Until a release is built, The .exe can be downloaded along with two small testing LAS pointclouds [here](https://drive.google.com/file/d/1py6crABTzHmLyuivs6GLZNGYrFZTVAhH/view?usp=sharing)

The GUI had the advantage of not needing any additional software however it does not handle errors well. At this stage, the user interface is extremely basic. 

## Python Command line Script

### Command Line Requirements ###

* Python \>3 (tested on 3.7, 3.8)
* GDAL Python bindings
* Numpy
* Scipy
* Laspy

## Setup ##

#### GDAL ####
Download GDAL wheel file matching python version and operating system [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal)

With the desired python interpreter/environment

```pip install "path_to_downloaded_wheel.whl"```

Test in python with

```from osgeo import gdal```

#### Numpy, Scipy, Laspy ####

The rest of the required modules should work with a simple pip call

```
pip install scipy
pip install numpy
pip install laspy
```

## Using main.py ##


### Required Arguments ###

The only required parameters are ```beforeLAS```, ```afterLAS```, ```resolution``` in that order.
```beforeLAS```, and ```afterLAS``` are both filepaths to LAS files. 

```resolution``` is the desired raster resolution, using the same units as the LAS files. 

A simple example would be:

```python main.py "path_to_before.las" "path_to_after.las" 0.01```

### Optional Arguments ###

```--outputDir``` Pass this argument the path of a directory to control where the created files will be placed. By default, files will be created in the same directory as the script. 

```--pvalue``` This argument sets the p-value used to filter the raw DoD raster. Default is 0.05

```--soilDensity``` Pass this argument a soil bulk density value in kg/m^3 to have the script estimate a mass of surface change from the changed volume. By default, a mass calculation will not take place. Assumes resolution is in meters.

```--outputRasters``` Just including this argument will output the intermediate rasters used in the t-test and DoD calculations. By default, intermediate rasters will not be created. 

```--maskingRaster``` Pass this argument the filepath of a binary .tif file to clip the DoD output to that raster. (0's representing exclusion, 1's representing inclusion)

### Outputs ###

| Output File        | Condition           | What is the file?  |
| ------------- |:-------------:| -----:|
| filter_dod.tif      | always | P-filtered DoD, not masked |
| raw_dod.tif      | always      |  Raw DoD, not masked |
| output.csv | always      |   Statistics from p-filtered DoD operation |
| p_score.tif | always      | Raster of calculated p-values |
| t_score.tif | always      | Raster of calculated t-scores |
| filter_dod_c.tif | if ```--maskingRaster``` is used      |   P-filtered DoD, Masked |
| raw_dod_c.tif | if ```--maskingRaster``` is used      |   Raw DoD, Masked |
| a_mean.tif, b_mean.tif | if ```--outputRasters``` is used      |   Mean cell height values from the (b)efore and (a)fter LAS files |
| a_count.tif, b_count.tif | if ```--outputRasters``` is used      | Count of points falling in each cell from the (b)efore and (a)fter LAS files |
| a_std.tif, b_std.tif | if ```--outputRasters``` is used      | Standard Deviation of cell height values from the (b)efore and (a)fter LAS files |

### Assumptions and Limitations ###

* LAS files need to at least be roughly aligned and using the same coordiante system / units.
* A good deal of memory may be needed for large files. In testing,  2 LAS files of a combined size of 5.75 GB and 150 Million+ points used a little more than 14 GB of RAM
* Only takes .las and .tif files. 

