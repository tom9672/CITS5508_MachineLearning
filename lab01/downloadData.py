import urllib.request
import tarfile
import os

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz"

DIR_PATH = os.path.join('lab01','datasets','housing')
FILE_PATH = os.path.join(DIR_PATH,'housing.tgz')

# Use urllib.request download data, save as housing.tgz
urllib.request.urlretrieve(DOWNLOAD_ROOT, FILE_PATH)

# extract tgz file, to get csv file
housing_tgz = tarfile.open(FILE_PATH)
housing_tgz.extractall(DIR_PATH)
housing_tgz.close()


