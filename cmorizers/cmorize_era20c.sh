#!/bin/env bash

ESGF=/scratch/yanchun/ESGF/
cp /projects/NS9588K/ERA20c/psl_mon_1900_2010.nc $ESGF/rawdata/obs
mv psl_mon_1900_2010.nc OBS6_ERA-20C_reanaly_1_Amon_psl_190001-201012.nc
ncrename -v sp,psl OBS6_ERA-20C_reanaly_1_Amon_psl_190001-201012.nc
ncrename -d latitude,lat -d longitude,lon OBS6_ERA-20C_reanaly_1_Amon_psl_190001-201012.nc
ncrename -v latitude,lat -v longitude,lon OBS6_ERA-20C_reanaly_1_Amon_psl_190001-201012.nc
ncatted -a standard_name,psl,m,c,air_pressure_at_mean_sea_level OBS6_ERA-20C_reanaly_1_Amon_psl_190001-201012.nc
ncatted -a long_name,psl,m,c,Sea Level Pressure OBS6_ERA-20C_reanaly_1_Amon_psl_190001-201012.nc
ncatted -a long_name,psl,m,c,"Sea Level Pressure" OBS6_ERA-20C_reanaly_1_Amon_psl_190001-201012.nc

