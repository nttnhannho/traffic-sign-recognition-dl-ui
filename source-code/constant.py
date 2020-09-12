import os

# current working path
CURRENT_WORKING_PATH = os.getcwd()

# form config
TITLE = "TRAFFIC SIGN RECOGNITION TOOL"
HEIGHT = 300
WIDTH = 600
ICON_PATH = CURRENT_WORKING_PATH + '\\icon\\gui_icon.ico'
TOPIC = "DEMO PROGRAM"

# image size
IMG_HEIGHT = 32
IMG_WIDTH = 32
# accepted Accuracy for result of prediction
RESULT_ACCURACY = 0.8

SIGNNAMES = CURRENT_WORKING_PATH + "\\input\\signnames.csv"

# image info
IMG_INFO = "You are choosing an image from test set."
# test set path
temp = CURRENT_WORKING_PATH + "/input/test"
IMG_TESTSET = temp.replace('\\', '/')

# result path
RESULT_PATH = CURRENT_WORKING_PATH + "\\result\\"
