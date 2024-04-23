# MPI_MiniProject
Mini project as part of my PG Computational Data Science

It implements various concepts on MPI (Message Passing Interface) and linear regression using normal equation

Install MPI using below command:
!pip install mpi4py

PowerPlantData.csv is the data set to work on which is attached in this repo.

In scatter_among_workers.py line 147 replace the correct path to this data file.

The scatter_among_workers.py can be run on MPI using below command (it'll start 3 MPI processes and divide the data among 3 and do the processing):

!mpirun --allow-run-as-root --oversubscribe -np 3 python scatter_among_workers.py
