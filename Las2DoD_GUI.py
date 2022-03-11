import numpy as np
from scipy import stats
import laspy
from osgeo import gdal, ogr
import os
import tkinter
from tkinter import filedialog
import sys
import base64
import threading


#  ~~~~~ Functions for GUI ~~~~~~~~~~~~
def on_entry_click(event, text):
    """function that gets called whenever entry is clicked"""
    if event.widget.get() == text:
        event.widget.delete(0, "end")  # delete all the text in the entry
        event.widget.insert(0, '')  # Insert blank for user input
        event.widget.config(fg='black')


def on_focusout(event, text):
    if event.widget.get() == '':
        event.widget.insert(0, text)
        event.widget.config(fg='grey')


# Select only LAS files
def browseFileNameLAS(inputVar):
    file = tkinter.filedialog.askopenfilename(filetypes=[("LAS files", "*.las")])
    inputVar.set(file)


# Select only Tif files
def browseFileNameTIF_SHP(inputVar):
    file = tkinter.filedialog.askopenfilename(filetypes=[("TIF files", "*.tif")])
    inputVar.set(file)


def browseDir(inputVar):
    file = tkinter.filedialog.askdirectory()
    inputVar.set(file)


def redirector(inputStr):
    Console.insert(tkinter.INSERT, inputStr)
    Console.update_idletasks()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Using gdal, Clips a raster with a shp, or with a raster via raster>shapefile
def ClipRasterByFile(raster_to_clip, masking_file, output_raster):
    try:
        raster_dsc = gdal.Open(masking_file)
        raster_band = raster_dsc.GetRasterBand(1)
    except RuntimeError as e:
        print(f'Error: {e}')
        sys.exit(1)

    output_shapefile = r"temp.shp"
    drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = drv.CreateDataSource(output_shapefile)
    dst_layer = dst_ds.CreateLayer(output_shapefile, srs=None)
    gdal.Polygonize(raster_band, raster_band, dst_layer, -1, [], callback=None)

    raster_dsc = None
    raster_band = None
    dst_ds = None
    dst_layer = None

    # Clip by shapefile
    out_tile = gdal.Warp(output_raster, raster_to_clip, cutlineDSName=output_shapefile, cropToCutline=True,
                         dstNodata=-9999)
    out_tile = None

    # Removing intermediate shapefile and associated files
    os.remove("temp.shp")
    os.remove("temp.dbf")
    os.remove("temp.shx")


# Getting Summary statistics from DoD
def sumDod(input_dod, soil_density=None):
    gdal_ds = gdal.Open(input_dod)
    band = gdal_ds.GetRasterBand(1)

    noDataValue = band.GetNoDataValue()

    # Get raster georeference info
    transform = gdal_ds.GetGeoTransform()
    pixelWidth = abs(transform[1])
    pixelHeight = abs(transform[5])

    array = band.ReadAsArray()
    np_array = np.array(array)

    sum_with_value = 0
    pos_sum_with_value = 0
    neg_sum_with_value = 0
    count = 0
    for index, cell in enumerate(np.nditer(np_array)):
        if cell != noDataValue:
            if cell > 0:
                sum_with_value += cell
                count += 1
                pos_sum_with_value += cell
            if cell < 0:
                sum_with_value += cell
                count += 1
                neg_sum_with_value += cell

    # change_over_slope_m = sum_with_value
    added_volume = round(pos_sum_with_value * (pixelWidth * pixelHeight), 4)
    subtracted_volume = round(neg_sum_with_value * (pixelWidth * pixelHeight), 4)
    area_sqr_m = round(count * pixelWidth * pixelHeight, 4)
    cube_m_change = round(sum_with_value * (pixelWidth * pixelHeight), 4)
    # cube_m_change = sum_with_value * area_sqr_m
    if soil_density:
        expected_sediment_kg = round(cube_m_change * soil_density, 4)
    else:
        expected_sediment_kg = None

    # Dereferencing inputs
    gdal_ds = None
    band = None
    return count, area_sqr_m, added_volume, subtracted_volume, cube_m_change, expected_sediment_kg


# Ceiling Division for creating grid
def ceildiv(a, b):
    return -(-a // b)


# Numpy array to raster
def array2Raster(array, fileName, xmin, resolution, ymax, n_columns, n_rows, overwrite=True):
    # Deleting file that is to be written
    if overwrite:
        if os.path.exists(fileName):
            os.remove(fileName)

    # GDAL Stuff
    geotransformation = (xmin, resolution, 0, ymax, 0, -resolution)
    output_raster = gdal.GetDriverByName('GTiff').Create(fileName, n_columns, n_rows, 1,
                                                         gdal.GDT_Float32)  # Open the file
    output_raster.SetGeoTransform(geotransformation)  # Specify its coordinates
    # srs = osr.SpatialReference()  # Establish its coordinate encoding
    # srs.ImportFromEPSG(4326)  # This one specifies WGS84 lat long.
    # output_raster.SetProjection(srs.ExportToWkt())  # Exports the coordinate system to the file
    output_raster.GetRasterBand(1).SetNoDataValue(-9999.0)
    output_raster.GetRasterBand(1).WriteArray(array)  # Writes my array to the raster
    output_raster.FlushCache()
    # output_raster.GetRasterBand(1).ComputeStatistics(False)
    output_raster = None


# Main program for GUI
def callback():
    # Setting main program parameters from GUI
    beforeLAS = Before_LAS_Entry.get()
    afterLAS = After_LAS_Entry.get()
    outputDir = Out_Dir_Entry.get()
    resolution = float(resolution_Entry.get())
    outputRasters = int_raster_box_var.get()  # Comes to 0, 1:
    pvalue = float(p_v_Entry.get())
    maskingRaster = Mask_Ras_Entry.get()
    if maskingRaster == 'Optional':  # This needs to match GUI default
        maskingRaster = None
    if Soil_density_Entry.get() == 'Optional (e.g. 1250)':  # This needs to match the GUI default
        soilDensity = None
    else:
        soilDensity = int(Soil_density_Entry.get())

    # Begin Main Program
    print('Reading LAS files..\n')

    b_las = laspy.read(beforeLAS)
    a_las = laspy.read(afterLAS)

    # Operating on scaled las values, potential rounding issues?
    b_coords = np.vstack((b_las.x, b_las.y, b_las.z)).transpose()
    a_coords = np.vstack((a_las.x, a_las.y, a_las.z)).transpose()

    xmax = (np.maximum(b_coords[:, 0].max(), a_coords[:, 0].max()))
    xmin = (np.minimum(b_coords[:, 0].min(), a_coords[:, 0].min()))
    ymax = (np.maximum(b_coords[:, 1].max(), a_coords[:, 1].max()))
    ymin = (np.minimum(b_coords[:, 1].min(), a_coords[:, 1].min()))

    b_x = b_coords[:, 0]
    b_y = b_coords[:, 1]
    b_z = b_coords[:, 2]
    b_coords = None

    a_x = a_coords[:, 0]
    a_y = a_coords[:, 1]
    a_z = a_coords[:, 2]
    a_coords = None

    print('Creating Grid...\n')

    # Making Arrays of lat and long, y and x  - I added one here just to cover rounding? it made it match CC
    n_rows = int((ceildiv(ymax - ymin, resolution))) + 1
    n_columns = int((ceildiv(xmax - xmin, resolution))) + 1

    print(f'Rows: {n_rows}')
    print(f'Columns: {n_columns}\n')

    # Operate off of arrays, no creating intermediate rasters
    print('Creating Numpy Arrays...\n')

    # Nan values are replaced, but not zeros for std, count
    b_z_mean = stats.binned_statistic_2d(b_x, b_y, b_z, statistic='mean', bins=(n_columns, n_rows))
    b_z_mean = np.flip(b_z_mean.statistic.transpose(), 0)
    np.nan_to_num(b_z_mean, copy=False, nan=-9999)

    b_z_std = stats.binned_statistic_2d(b_x, b_y, b_z, statistic='std', bins=(n_columns, n_rows))
    b_z_std = np.flip(b_z_std.statistic.transpose(), 0)
    np.nan_to_num(b_z_std, copy=False, nan=-9999)

    b_z_count = stats.binned_statistic_2d(b_x, b_y, b_z, statistic='count', bins=(n_columns, n_rows))
    b_z_count = np.flip(b_z_count.statistic.transpose(), 0)
    np.nan_to_num(b_z_count, copy=False, nan=-9999)

    a_z_mean = stats.binned_statistic_2d(a_x, a_y, a_z, statistic='mean', bins=(n_columns, n_rows))
    a_z_mean = np.flip(a_z_mean.statistic.transpose(), 0)
    np.nan_to_num(a_z_mean, copy=False, nan=-9999)

    a_z_std = stats.binned_statistic_2d(a_x, a_y, a_z, statistic='std', bins=(n_columns, n_rows))
    a_z_std = np.flip(a_z_std.statistic.transpose(), 0)
    np.nan_to_num(a_z_std, copy=False, nan=-9999)

    a_z_count = stats.binned_statistic_2d(a_x, a_y, a_z, statistic='count', bins=(n_columns, n_rows))
    a_z_count = np.flip(a_z_count.statistic.transpose(), 0)
    np.nan_to_num(a_z_count, copy=False, nan=-9999)

    # Create Rasters of the arrays
    if outputRasters:
        array2Raster(b_z_mean, rf'{outputDir}\b_mean.tif', xmin, resolution, ymax, n_columns, n_rows)
        array2Raster(b_z_std, rf'{outputDir}\b_std.tif', xmin, resolution, ymax, n_columns, n_rows)
        array2Raster(b_z_count, rf'{outputDir}\b_count.tif', xmin, resolution, ymax, n_columns, n_rows)
        array2Raster(a_z_mean, rf'{outputDir}\a_mean.tif', xmin, resolution, ymax, n_columns, n_rows)
        array2Raster(a_z_std, rf'{outputDir}\a_std.tif', xmin, resolution, ymax, n_columns, n_rows)
        array2Raster(a_z_count, rf'{outputDir}\a_count.tif', xmin, resolution, ymax, n_columns, n_rows)

    # Creating output arrays - Currently based on the shape of the before
    out_t_array = np.zeros(b_z_mean.shape, b_z_mean.dtype)
    out_p_array = np.zeros(b_z_mean.shape, b_z_mean.dtype)
    out_dod_array = np.zeros(b_z_mean.shape, b_z_mean.dtype)
    out_fdod_array = np.zeros(b_z_mean.shape, b_z_mean.dtype)

    print('Running t-tests...\n')

    # Running t_test from arrays
    for mrb, prb, srb, mra, pra, sra, outt, outp, outd, outf in np.nditer([b_z_mean, b_z_count, b_z_std,
                                                                           a_z_mean, a_z_count, b_z_std,
                                                                           out_t_array, out_p_array, out_dod_array,
                                                                           out_fdod_array],
                                                                          op_flags=['readwrite']):
        if mrb == -9999 or mra == -9999:
            outt[...] = -9999.0
            outp[...] = -9999.0
            outd[...] = -9999.0
            outf[...] = -9999.0
        else:
            test_1, test_2 = stats.ttest_ind_from_stats(mrb, srb, prb, mra, sra, pra, equal_var=False)

            if np.isinf(test_1):  # sometimes it gets infinity?
                outt[...] = -9999
            else:
                outt[...] = round(test_1, 6)
                outp[...] = round(test_2, 6)
                outd[...] = (mra - mrb)
        if outp[...] <= pvalue:
            outf[...] = outd[...]
        else:
            outf[...] = -9999.0

    print('Writing t-score tifs...\n')

    array2Raster(out_t_array, rf'{outputDir}\t_score.tif', xmin, resolution, ymax, n_columns, n_rows)
    out_t_array = None
    array2Raster(out_p_array, rf'{outputDir}\p_score.tif', xmin, resolution, ymax, n_columns, n_rows)
    out_p_array = None
    array2Raster(out_dod_array, rf'{outputDir}\raw_dod.tif', xmin, resolution, ymax, n_columns, n_rows)
    out_dod_array = None
    array2Raster(out_fdod_array, rf'{outputDir}\filter_dod.tif', xmin, resolution, ymax, n_columns, n_rows)
    out_fdod_array = None

    if maskingRaster:
        print('Clipping Rasters...\n')

        ClipRasterByFile(rf'{outputDir}\filter_dod.tif', maskingRaster,
                         rf'{outputDir}\filter_dod_c.tif')
        ClipRasterByFile(rf'{outputDir}\raw_dod.tif', maskingRaster, rf'{outputDir}\raw_dod_c.tif')
        final_file_name = 'filter_dod_c.tif'
    else:
        final_file_name = 'filter_dod.tif'

    print('Computing Volume Change...\n')

    count, area_sqr_m, added_volume, subtracted_volume, \
    cube_m_change, expected_sediment_kg = sumDod(rf'{outputDir}\{final_file_name}', soilDensity)

    headers = ['before_las', 'after_las', 'masking_file', 'p_value', 'resolution', 'soil_density_kgm3',
               'cell_count', 'area_sqr_m', 'added_volume', 'subtracted_volume', 'cube_m_change',
               'expected_sediment_kg']

    # Since files will be written without importing another module, need to make sure they are strings
    out_file = [str(beforeLAS), str(afterLAS), str(maskingRaster), str(pvalue), str(resolution),
                str(soilDensity),
                str(count), str(area_sqr_m), str(added_volume), str(subtracted_volume), str(cube_m_change),
                str(expected_sediment_kg)]

    print('Writing file...\n')

    with open(rf'{outputDir}\output.csv', 'w') as of:
        of.write(','.join(headers))
        of.write('\n')
        of.write(','.join(out_file))
        of.write('\n')

    print('Done')


# ~~~~~~~~~~~~~~~~~~~~ Icon Area ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##The Base64 icon version as a string
icon = \
    """
    Qk04AgAAAAAAADYAAAAoAAAAEAAAABAAAAABABAAAAAAAAICAAASCwAAEgsAAAAAAAAAAAAA/3//
    f/9//3/wQ+AD4APgA+AD4gsfAB8AHwAfAB8AXwj/f/9//3//f/BD4APgA+AD4APiCx8AHwAfAB8A
    HwBfCP9//3//f/9/8EPgA+AD4APgA+ILHwAfAB8AHwAfAF8I/3//f/9//3/wQ+AD4APgA+AD4gsf
    AB8AHwAfAB8AXwj/f/9//3//f/BD4APgA+AD4APiC58xnzGfMZ8xnzG/Nf9/e3+cf/9//3//f/9/
    /3//f/9/QnwAfAB8AHwAfAB8/38xfrV+/3//f/9//3//f/9//39CfAB8AHwAfAB8AHz/f/9//3//
    fx9jU3oxfv9//3//f0J8AHwAfAB8AHwAfP9/+2/6a/9/fy0ScjF+/3//f/9/QnwAfAB8AHwAfAB8
    /3/WV/NL/3//f/9//3//f/9//39CfAB8AHwAfAB8AHz/f98Yf2//f/9//3//f/9//3//f/9//3//
    f/9//3//f/9//38Yf2t9937/f/9//3//f781v3f/f/9//3//f/9//3//fxh/c369f/9/91/tN/9/
    v1bfe/9//3//f/9//3//f/9//nv/f/9/flJdSvxz/3//f/9//38fY19r/3//f/9/+WfoI19r/z03
    RTk9/3/Wfmt9/3//f58xX0r/f/9//3//f/9/v3cfY/9//3//f/9/3n//f/9//15/b/9//38AAA==
    """

icondata = base64.b64decode(icon)
# The temp file is icon.ico
tempFile = rf"{os.getcwd()}\icon.ico"
iconfile = open(tempFile, "wb")
# Extract the icon
iconfile.write(icondata)
iconfile.close()

# ~~~~~~~~~~~~~~~~~~~~~~~ Main GUI Area ~~~~~~~~~~~~~~~~~~~~~~~~~~~
root = tkinter.Tk()
root.tk.call('tk', 'scaling', 2.0)
root.title('Las2DoD')

try:
    root.wm_iconbitmap(tempFile)
except Exception as ex:
    print(f"An exception of type {type(ex).__name__} occurred. Arguments:\n{ex.args}")
    print('Skipping setting an icon')

## Delete the tempfile
os.remove(tempFile)

root.grid()

# Before LAS
Before_LAS_Var = tkinter.StringVar()
Before_LAS_Label = tkinter.Label(root, text='Before LAS: ').grid(row=1, column=0, pady=5, padx=5, sticky='E')
Before_LAS_Entry = tkinter.Entry(root, textvariable=Before_LAS_Var, width=50)
Before_LAS_text = os.getcwd()  # Should make this the working directory
Before_LAS_Entry.insert(0, Before_LAS_text)
Before_LAS_Entry.bind('<FocusIn>', lambda event, arg=Before_LAS_text: on_entry_click(event, arg))
Before_LAS_Entry.bind('<FocusOut>', lambda event, arg=Before_LAS_text: on_focusout(event, arg))
Before_LAS_Entry.config(fg='grey')
Before_LAS_Entry.grid(row=1, column=1, pady=5, padx=5)
Before_LAS_s_browse_button = tkinter.Button(root, text='Browse',
                                            command=lambda: browseFileNameLAS(Before_LAS_Var)).grid(row=1, column=2,
                                                                                                    pady=5, padx=20,
                                                                                                    sticky='W')

# After LAS
After_LAS_Var = tkinter.StringVar()
After_LAS_Label = tkinter.Label(root, text='After LAS: ').grid(row=2, column=0, pady=5, padx=5, sticky='E')
After_LAS_Entry = tkinter.Entry(root, textvariable=After_LAS_Var, width=50)
After_LAS_text = os.getcwd()  # Should make this the working directory
After_LAS_Entry.insert(0, After_LAS_text)
After_LAS_Entry.bind('<FocusIn>', lambda event, arg=After_LAS_text: on_entry_click(event, arg))
After_LAS_Entry.bind('<FocusOut>', lambda event, arg=After_LAS_text: on_focusout(event, arg))
After_LAS_Entry.config(fg='grey')
After_LAS_Entry.grid(row=2, column=1, pady=5, padx=5)
After_LAS_s_browse_button = tkinter.Button(root, text='Browse',
                                           command=lambda: browseFileNameLAS(After_LAS_Var)).grid(row=2, column=2,
                                                                                                  pady=5, padx=20,
                                                                                                  sticky='W')

# Output Directory
Out_Dir_Var = tkinter.StringVar()
Out_Dir_Label = tkinter.Label(root, text='Output Directory: ').grid(row=3, column=0, pady=5, padx=5, sticky='E')
Out_Dir_Entry = tkinter.Entry(root, textvariable=Out_Dir_Var, width=50)
Out_Dir_text = os.getcwd()  # Should make this the working directory
Out_Dir_Entry.insert(0, Out_Dir_text)
Out_Dir_Entry.bind('<FocusIn>', lambda event, arg=Out_Dir_text: on_entry_click(event, arg))
Out_Dir_Entry.bind('<FocusOut>', lambda event, arg=Out_Dir_text: on_focusout(event, arg))
Out_Dir_Entry.config(fg='grey')
Out_Dir_Entry.grid(row=3, column=1, pady=5, padx=5)
Out_Dir_s_browse_button = tkinter.Button(root, text='Browse',
                                         command=lambda: browseDir(Out_Dir_Var)).grid(row=3, column=2, pady=5, padx=20,
                                                                                      sticky='W')

# Masking Raster
Mask_Ras_Var = tkinter.StringVar()
Mask_Ras_Label = tkinter.Label(root, text='Masking Raster: ').grid(row=4, column=0, pady=5, padx=5, sticky='E')
Mask_Ras_Entry = tkinter.Entry(root, textvariable=Mask_Ras_Var, width=50)
Mask_Ras_text = 'Optional'
Mask_Ras_Entry.insert(0, Mask_Ras_text)
Mask_Ras_Entry.bind('<FocusIn>', lambda event, arg=Mask_Ras_text: on_entry_click(event, arg))
Mask_Ras_Entry.bind('<FocusOut>', lambda event, arg=Mask_Ras_text: on_focusout(event, arg))
Mask_Ras_Entry.config(fg='grey')
Mask_Ras_Entry.grid(row=4, column=1, pady=5, padx=5)
Mask_Ras_s_browse_button = tkinter.Button(root, text='Browse',
                                          command=lambda: browseFileNameTIF_SHP(Mask_Ras_Var)).grid(row=4, column=2,
                                                                                                    pady=5, padx=20,
                                                                                                    sticky='W')

# Soil Bulk Density
Soil_density_Var = tkinter.StringVar()
Soil_density_Label = tkinter.Label(root, text='Soil Bulk Density in g/m^3: ').grid(row=5, column=0, pady=5, padx=5,
                                                                                   sticky='E')
Soil_density_Entry = tkinter.Entry(root, textvariable=Soil_density_Var, width=50)
Soil_density_text = 'Optional (e.g. 1250)'  # Should make this the working directory
Soil_density_Entry.insert(0, Soil_density_text)
Soil_density_Entry.bind('<FocusIn>', lambda event, arg=Soil_density_text: on_entry_click(event, arg))
Soil_density_Entry.bind('<FocusOut>', lambda event, arg=Soil_density_text: on_focusout(event, arg))
Soil_density_Entry.config(fg='grey')
Soil_density_Entry.grid(row=5, column=1, pady=5, padx=5)

# p-value
p_v_Var = tkinter.StringVar()
p_v_Label = tkinter.Label(root, text='P-value: ').grid(row=6, column=0, pady=5, padx=5, sticky='E')
p_v_Entry = tkinter.Entry(root, textvariable=p_v_Var, width=25)
p_v_text = '0.05'  # Should make this the working directory
p_v_Entry.insert(0, p_v_text)
p_v_Entry.bind('<FocusIn>', lambda event, arg=p_v_text: on_entry_click(event, arg))
p_v_Entry.bind('<FocusOut>', lambda event, arg=p_v_text: on_focusout(event, arg))
p_v_Entry.config(fg='grey')
p_v_Entry.grid(row=6, column=1, pady=5, padx=5)

# Resolution
resolution_Var = tkinter.StringVar()
resolution_Label = tkinter.Label(root, text='Resolution: ').grid(row=7, column=0, pady=5, padx=5, sticky='E')
resolution_Entry = tkinter.Entry(root, textvariable=resolution_Var, width=25)
resolution_text = '0.01'  # Should make this the working directory
resolution_Entry.insert(0, resolution_text)
resolution_Entry.bind('<FocusIn>', lambda event, arg=resolution_text: on_entry_click(event, arg))
resolution_Entry.bind('<FocusOut>', lambda event, arg=resolution_text: on_focusout(event, arg))
resolution_Entry.config(fg='grey')
resolution_Entry.grid(row=7, column=1, pady=5, padx=5)

# int_raster raster option
int_raster_label = tkinter.Label(root, text='Output Intermediate Rasters?').grid(row=8, column=1, pady=5, padx=5)
int_raster_box_var = tkinter.IntVar()
int_raster_box = tkinter.Checkbutton(root, variable=int_raster_box_var)
int_raster_box.grid(row=8, column=2, pady=5, padx=5, sticky='W')

# Run button
def thread_run():
    threading.Thread(target=callback).start()


Button = tkinter.Button(root, text='Run', command=thread_run, width=15).grid(row=9, column=1, columnspan=1, pady=5,
                                                                           padx=20)


# Writing console output to GUI box
console_label = tkinter.Label(root, text='Console Output:').grid(row=10, column=1, columnspan=1, pady=5, padx=20)
Console = tkinter.Text(root)
Console.grid(row=11, column=0, columnspan=3, pady=5, padx=20)

# Writing console to GUI
sys.stdout.write = redirector

root.mainloop()