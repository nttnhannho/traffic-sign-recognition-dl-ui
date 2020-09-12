import os
import numpy as np 
import cv2
import constant as cons
from PIL import Image

def get_roi(path, x1_val, y1_val, x2_val, y2_val):
    try:
        slash_split = path.rpartition('/')
        x = slash_split[len(slash_split) - 1]
        dot_pos = x.rfind('.')
        int_p = int(x[:dot_pos])
        x1 = x1_val[int_p]
        y1 = y1_val[int_p]
        x2 = x2_val[int_p]
        y2 = y2_val[int_p]
        
        return x1, y1, x2, y2
    except Exception as ex:
        print(ex)

def create_directory(path):
    try:
        if not os.path.exists(path):
            os.mkdir(path)
    except Exception as ex:
        print(ex)
        
def make_center(root):
    try:
        root.focus_force()
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    except Exception as ex:
        print(ex)
    
def check_valid_path(path):
    try:
        if os.path.isfile(path) and os.path.exists(path):
            return True
        
        return False
    except Exception as ex:
        print(ex)

# use OpenCV to find Contours. Base on Contours we define is there any traffic signs in the image.
def preprocess_img(imgRGB, erode_dilate=True):
    """Preprocess the image for contour detection.
    Args:
        imgBGR: source image.
        erode_dilate: erode and dilate or not.
    Return:
        img_bin: a binary image (blue and red).
    """
    try:
        rows, cols, _ = imgRGB.shape
        imgHSV = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2HSV)
    
        lower_blue = np.array([100, 43, 46])
        upper_blue = np.array([124, 255, 255])
        mask_blue = cv2.inRange(imgHSV, lower_blue, upper_blue)
        
        lower_red1 = np.array([0, 43, 46])
        upper_red1 = np.array([10, 255, 255])
        mask_red1 = cv2.inRange(imgHSV, lower_red1, upper_red1)
        
        lower_red2 = np.array([156, 43, 46])
        upper_red2 = np.array([180, 255, 255])
        mask_red2 = cv2.inRange(imgHSV, lower_red2, upper_red2)
        
        
        mask_red = np.maximum(mask_red1, mask_red2)
        
        img_bin = np.maximum(mask_blue, mask_red)
    
        if erode_dilate is True:
            kernelErosion = np.ones((9,9), np.uint8)
            kernelDilation = np.ones((9,9), np.uint8)
            img_bin = cv2.erode(img_bin, kernelErosion, iterations=2)
            img_bin = cv2.dilate(img_bin, kernelDilation, iterations=2)
    
        return img_bin
    except Exception as ex:
        print(ex)

def detect_contour(img_bin, min_area=0, max_area=-1, wh_ratio=2.0):
    """Detect contours in a binary image.
    Args:
        img_bin: a binary image.
        min_area: the minimum area of the contours detected.
            (default: 0)
        max_area: the maximum area of the contours detected.
            (default: -1, no maximum area limitation)
        wh_ratio: the ration between the large edge and short edge.
            (default: 2.0)
    Return:
        rects: a list of rects enclosing the contours. if no contour is detected, rects=[]
    """
    try:
        rects = []
        contours, _ = cv2.findContours(img_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if len(contours) == 0:
            return rects
    
        max_area = img_bin.shape[0]*img_bin.shape[1] if max_area < 0 else max_area
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area and area <= max_area:
                x, y, w, h = cv2.boundingRect(contour)
                if 1.0*w/h < wh_ratio and 1.0*h/w < wh_ratio:
                    rects.append([x, y, w, h])
                    
        return rects
    except Exception as ex:
        print(ex)

def draw_rects_on_img(img, rects):
    """ draw rects on an image.
    Args:
        img: an image where the rects are drawn on.
        rects: a list of rects.
    Return:
        img_rects: an image with rects.
    """
    try:
        img_copy = img.copy()
        for rect in rects:
            x, y, w, h = rect
            cv2.rectangle(img_copy, (x,y), (x+w,y+h), (0,255,0), 2)
            
        return img_copy
    except Exception as ex:
        print(ex)
        
def recognize_sign(images, model):
    """ Use model train before predict image.
    Args:
        images: traffic signs after detection
    Return:
        pred: classes of the traffic signs
    """
    try:
        data = []
        for image in images:
            image_from_array = Image.fromarray(image, 'RGB')
            size_image = image_from_array.resize((cons.IMG_HEIGHT, cons.IMG_WIDTH))
            data.append(np.array(size_image))
        X_test = np.array(data)
        X_test = X_test.astype('float32')/255
        pred = [-1]
        if np.amax(model.predict(X_test)) >= cons.RESULT_ACCURACY:
            pred = model.predict_classes(X_test)
            
        return pred
    except Exception as ex:
        print(ex)

def load_name(index, label_values):
    """ Return name of the traffic sign's class
    Args:
        index: class id
    Return:
        Name of the traffic sign's class
    """
    try:
        if index == -1:
            return "Unidentified"
        
        return label_values[index]
    except Exception as ex:
        print(ex)
