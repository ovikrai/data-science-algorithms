# Procedures Imports
import math
from matplotlib import pyplot
import numpy
import scipy
from src.data_science.etl.data_extractor import DataExtractor


# Structures Imports
import platform

print("######################################## \n")
print("########## START MAIN CLIENT ########## \n")
print("########## SYSTEM INFORMATION ##########")
print("########## ARCHITECTURE:", platform.processor())
print("########## OPERATING SYSTEM:", platform.system())
print("######################################## \n")

data = DataExtractor(
    "https://data.wa.gov/api/views/f6w7-q2d2/rows.csv?accessType=DOWNLOAD",
    "./output.csv",
)

data.download_dataset()
data.read_dataset()
data.format_columns()
data.create_columns()
data.delete_columns()


print("######################################## \n")
print("########## END MAIN CLIENT ##########")
print("######################################## \n")
