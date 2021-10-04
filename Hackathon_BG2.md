# NICEST2 D.D2: ESMvaltool more emphasis on nordic/arctic regions (group 2 at hackathon)  
- What do I expect a collection of Nordic-focused analysis scripts to contain?
- Which (more or less) Nordic-specific analyses do I or my colleagues perform? Which tools do I use to accomplish them?
- Are there any specific observational datasets required?

## general thoughts:
* is there a group working on the implementation for the ESMValTool for the nordic regions? 
  * All suggestions are welcome
  * good to have some semi-finished scripts or variables of interest. Time frame: Delivery by month 30 of NICEST2 project (mid 2022?)

* our research interests vs Nordic regions

* foster Nordic collaboration (does probably not need to be reseach on the Nordic regions). Getting people running ESMValTool will help on the collabaration regardless of the "Nordic". How to continue? How to start develop? Most of us have little experience.

* Is the NorESM INES comunity involved in this task? Maybe some from INES can start working on the recipe and the nordic seas region and so on, it will be very helpful

* still useful because it handles the data so well. so many convenient things. Also possible to use ESMValTool without sharing the recipe

* maybe start using ESMValTool for new analysis (not what I have already looked at). If you have to write a new script anyway. This lowers the threshold for people to start using it.

* a great tool also for students, a very good opportunity for universities to collaborate with other research groups like met offices

* How do we get people started? Communication channel , new hackaton or some kind of low threshold FAQ/collaboration, new hackaton, slack, 

## ideas and such


- get a quick overview of the data

- focusing on land and land - atmosphere interactions. Climate extremes e.g. rain on snow,  drought defined by soil moisture.  

- Snow and vegetation (e.g., LAI, GPP) related analysis will be useful. 

- Nordic specific observation dataset on flux measurments which can be included, and validation from other model simulations. 

- a good analysis of AMOC (not only maximum but meridional sections), mass transport across key domains (Denmark strait, Bering Strait +++), and transport of heat, salt and other tracers, historical trends of the last 20 years (temperature, mixed layer depth), process based evaluations (EOFs) what is the dominant mode of variability and how that is related to NAO,  Nordic Seas observations are sparse but the time series often long, monthly observations

- working on cloud feedback in the extra tropics and emergent constraints, 

- selecting models for downscaling, polar regions, NAO, geopotential heights index, many oscillations and modes -> circulation indices, scandinavian blocking and blocking in general

- decadal prediction, initialize ocean and sea ice, AMOC, many ensembles, large assessments, random selection and statistical tools for decadal predictions, obs data: sea-ice concentration, thickness, cloud data

- sea level rise, looking at both big ice sheet, Southern Ocean circulation interesting. Also East African monsoon

- aerosol description, look at the globe and not only nordic. 

- AOD measurments, 

- Can we use data from regional models for comparison? 
  It should be possible as long as CF-conventions are used.

- compare models and compare to observations, not so much going into detail in one model

- ESMValTool provides a good documentation, possible to share recipes, but not yet possible to use on NorESM raw data

- analysis output from NorESM, EC-EARTH, MPI, easy to adapt ESMValTool to all models

## Obstacles:

* observations from sites can be problematic (methodologically, not technically)

* can reformat the data (obs? or something else?) to cmorized standard. Nice to have a common dataset which we all can use, 

* not too much aerosol stuff in the cmip6 ensemble (e.g. size distribution), 

* need cmorized data? NorESM diagnostics -> generate a lot of figures fast. ESMValTool better for comparing data later on. When we are not in production mode, cmorizing data does not make sense

## Technical:

* is it possible to put an analysis script in the beginning (e.g. a randomizer) or just in the end in the recipe ESMValTool?
  * In general at the end, but you can use the output of one diagnostic as the input to another via so-called ancestors ( https://docs.esmvaltool.org/projects/esmvalcore/en/latest/recipe/overview.html#ancestor-tasks )

* quality control on ESMValTool recipes (is it correct?/are new diagnostics being checked by someone else other than the author). Oskar: The name of the author is defined, and is probably also responsible.
  *Integration of new analysis always via pull request. Code review from 1) technical point of view 2) from scientific point of view. No big masterplan of priority for new recipies. Usually happen because someone wants to do it.

