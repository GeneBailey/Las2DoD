import argparse
import os
import numpy as np
from scipy import stats
import laspy
from osgeo import gdal, ogr

# Using gdal, creates .shp from raster then uses that to clip another raster
def ClipRasterByRaster(raster_to_clip, masking_raster, output_raster):
    # Raster to shapefile
    raster_dsc = gdal.Open(masking_raster)
    raster_band = raster_dsc.GetRasterBand(1)

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

    # Removing intermediate shapefile
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

    #change_over_slope_m = sum_with_value
    added_volume = round(pos_sum_with_value * (pixelWidth * pixelHeight), 4)
    subtracted_volume = round(neg_sum_with_value * (pixelWidth * pixelHeight), 4)
    area_sqr_m = round(count * pixelWidth * pixelHeight, 4)
    cube_m_change = round(sum_with_value * (pixelWidth * pixelHeight), 4)
    #cube_m_change = sum_with_value * area_sqr_m
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
    #srs = osr.SpatialReference()  # Establish its coordinate encoding
    #srs.ImportFromEPSG(4326)  # This one specifies WGS84 lat long.
    #output_raster.SetProjection(srs.ExportToWkt())  # Exports the coordinate system to the file
    output_raster.GetRasterBand(1).SetNoDataValue(-9999.0)
    output_raster.GetRasterBand(1).WriteArray(array)  # Writes my array to the raster
    output_raster.FlushCache()
    #output_raster.GetRasterBand(1).ComputeStatistics(False)
    output_raster = None


# Function for Python command line
# Making sure the input LAS files exist and have the las file extension
def checkLas(f):
    if not os.path.exists(f):
        raise argparse.ArgumentTypeError(f"{f} does not exist.")
    if not f.endswith('.las'):
        raise argparse.ArgumentTypeError(f"input file must be a .las file")
    return f


def main(beforeLAS, afterLAS, resolution, outputRasters, outputDir, pvalue, soilDensity, maskingRaster):
    print(f'Before LAS: {beforeLAS}')
    print(f'After LAS: {afterLAS}')
    print(f'Resolution: {resolution}')
    print(f'Output Rasters?: {outputRasters}')
    print(f'Output Dir: {outputDir}')
    print(f'P-value: {pvalue}')
    print(f'Soil Bulk Density: {soilDensity}')
    print(f'Masking Raster: {maskingRaster}')

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

    # Binning off of number of columns, rows. Potential shifting if range of before, after clouds is different? Different start and end points?


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

    # Create Rasters of the arrays if option is elected
    if outputRasters:
        array2Raster(b_z_mean, rf'{outputDir}\b_mean.tif', xmin, resolution, ymax, n_columns, n_rows)
        array2Raster(b_z_std, rf'{outputDir}\b_std.tif', xmin, resolution, ymax, n_columns, n_rows)
        array2Raster(b_z_count, rf'{outputDir}\b_count.tif', xmin, resolution, ymax, n_columns, n_rows)
        array2Raster(a_z_mean, rf'{outputDir}\a_mean.tif', xmin, resolution, ymax, n_columns, n_rows)
        array2Raster(a_z_std, rf'{outputDir}\a_std.tif', xmin, resolution, ymax, n_columns, n_rows)
        array2Raster(a_z_count, rf'{outputDir}\a_count.tif', xmin, resolution, ymax, n_columns, n_rows)

    # Creating output arrays - Currently based on the shape of the before mean
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

        ClipRasterByRaster(rf'{outputDir}\filter_dod.tif', maskingRaster,
                           rf'{outputDir}\filter_dod_c.tif')
        ClipRasterByRaster(rf'{outputDir}\raw_dod.tif', maskingRaster, rf'{outputDir}\raw_dod_c.tif')
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LAS to DoD with uncertainty testing')
    parser.add_argument("beforeLAS", help="Filepath to LAS file representing before", type=checkLas)
    parser.add_argument("afterLAS", help="Filepath to LAS file representing after", type=checkLas)
    parser.add_argument("resolution", help="Desired resolution in the same units as LAS files", type=float)
    parser.add_argument("--outputRasters", help="Option to output intermediate rasters", action="store_true")
    parser.add_argument("--outputDir", help="Path to directory where files will be created", type=str,
                        default=os.path.dirname(os.path.realpath(__file__)))
    parser.add_argument("--pvalue", help="P-value used to filter DoD", type=float, default=0.05)
    parser.add_argument("--soilDensity", help="Soil bulk density in kg/m3", type=int)
    parser.add_argument("--maskingRaster", help="Filepath to Raster used to mask DoDs", type=str)
    args = parser.parse_args()
    main(args.beforeLAS, args.afterLAS, args.resolution, args.outputRasters, args.outputDir, args.pvalue,
         args.soilDensity, args.maskingRaster)