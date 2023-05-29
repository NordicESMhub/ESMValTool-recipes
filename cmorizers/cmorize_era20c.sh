#!/bin/env bash
# Script to make the observation data ERA-20C CF-compliant
# ERA-20C:
#   https://climatedataguide.ucar.edu/climate-data/era-20c-ecmwfs-atmospheric-reanalysis-20th-century-and-comparisons-noaas-20cr
# CF convention:
#   http://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html
#   and search in the Standard Names "air_pressure_at_mean_sea_level"
#
# Yanchun He, 22nd May, 2023

# copy rawobs
mkdir -p ../Data/rawobs/ && cd ../Data/rawobs/
wget http://ns9560k.web.sigma2.no/diagnostics/esmvaltool/yanchun/Data/rawobs/psl_mon_1900_2010.nc
cp psl_mon_1900_2010.nc ../Data/ESGF/obsdata/Tier3/ERA-20C/ 
cd ../Data/ESGF/obsdata/Tier3/ERA-20C/

# rename file
mv psl_mon_1900_2010.nc OBS6_ERA-20C_reanaly_1_Amon_psl_190001-201012.nc

# rename variable
ncrename -v sp,psl OBS6_ERA-20C_reanaly_1_Amon_psl_190001-201012.nc

# rename dimensions
ncrename -d latitude,lat -d longitude,lon OBS6_ERA-20C_reanaly_1_Amon_psl_190001-201012.nc
ncrename -v latitude,lat -v longitude,lon OBS6_ERA-20C_reanaly_1_Amon_psl_190001-201012.nc

# rename standard_name
ncatted -a standard_name,psl,m,c,air_pressure_at_mean_sea_level OBS6_ERA-20C_reanaly_1_Amon_psl_190001-201012.nc

# rename long_name
ncatted -a long_name,psl,m,c,"Sea Level Pressure" OBS6_ERA-20C_reanaly_1_Amon_psl_190001-201012.nc

