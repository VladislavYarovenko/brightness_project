import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from scipy import ndimage
from xml.etree import ElementTree

def lengthl(x1, y1, x2, y2):                          #function to calculate the length of the line
        return math.sqrt((x2-x1)**2 + (y2-y1)**2)

def extract_features(filename):
    '''Here, we extract the seven selected features from the input image'''
    
    img = cv2.imread(filename, 3)
                   
    #Extracting the bounding box of the main object for area by perim and aspect ratio
    #newly added - was not present in last year's file, were the area by perima and aspect ratio of the whole image was taken
    xml_directory = "images/val/xml"

    file_path = xml_directory + '\\' + filename[-28:-5] + '.xml'
    
    if filename.startswith('balanced_images'):
        file_path = xml_directory + '\\' + filename[-28:-5] + '.xml'
    tree = ElementTree.parse(file_path)
    root = tree.getroot()
    objects = root.findall('object')
    area_by_perim = 0
    aspect_ratio = 0
    area = 0
    perim = 0
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp_surf = 0
    edge_length1 = 0
    for object in objects:
        x_min = int(object.find("bndbox/xmin").text)
        x_max = int(object.find("bndbox/xmax").text)
        y_min = int(object.find("bndbox/ymin").text)
        y_max = int(object.find("bndbox/ymax").text)
        width = x_max - x_min
        height = y_max - y_min
        area = (width) * (height)
        perim = (width + height) * 2
        area_by_perim += area/perim
        aspect_ratio += width / height
        img_crop = img[y_min:y_max, x_min:x_max]
        img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
        # surf = cv2.xfeatures2d.SURF_create() 
        surf = cv2.ORB_create() 
        keypoints_surf = surf.detect(img_gray, None)
        kp_surf += len(keypoints_surf)
        
        
        if (img_crop.shape[0] > img_crop.shape[1]):
            p = img_crop.shape[0]
        else:
            p = img_crop.shape[1]
        minLine = 0.01*p
        gap = 0.1*p
        threshold = 10

        
        edges = cv2.Canny(img_gray, 50, 100) #The Canny algorithm
    #     img_gaussian = cv2.GaussianBlur(gray,(3,3),0)
    #     kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    #     kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    #     img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
    #     img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
    #     edges = img_prewittx + img_prewitty
        lines = cv2.HoughLinesP(edges,rho=1,theta=np.pi/180, threshold=threshold, minLineLength = minLine, maxLineGap = gap)

        fs = []
        if lines is None:
            edge_length1 = 0
        else:
            for l in lines:
                for x1,y1,x2,y2 in l:
                    fs.append(lengthl(x1,y1,x2,y2))
            ls = [i/p for i in fs] #normalize to image length
            lhist = np.histogram(ls, bins = 7, range=(0,1))
            edge_length1 += lhist[0].tolist()[0] + lhist[0].tolist()[1] + lhist[0].tolist()[2]
        

    aspect_ratio = aspect_ratio / len(objects)
    edge_length1 = edge_length1/len(objects)
    # aspect_ratio = float(img.shape[1]) / img.shape[0]
    #
    # area_by_perim = img.shape[0] * img.shape[1] / ((img.shape[0] + img.shape[1]) * 2)              # area by perimeter
    
    
    
    '''Average perceived brightness'''
    B = np.mean(img[:, :, 0])
    G = np.mean(img[:, :, 1])
    R = np.mean(img[:, :, 2])
    average_perceived_brightness = (math.sqrt(0.241*(R**2) + 0.691*(G**2) + 0.068*(B**2)))  # average percevied brightness
    #average_perceived_brightness = 0.299*(R) + 0.587*(G) + 0.114*(B)
    #(0.299*R + 0.587*G + 0.114*B)

    
    
    
#     '''Edges - simplecv method - newly added'''
#     #http://bennycheung.github.io/recognizing-snacks-using-simplecv
# #     img_2 = img[y_min:y_max, x_min:x_max]
#     if (img.shape[0] > img.shape[1]):
#         p = img.shape[0]
#     else:
#         p = img.shape[1]
#     minLine = 0.01*p
#     gap = 0.1*p
#     threshold = 10

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, 50, 100) #The Canny algorithm
# #     img_gaussian = cv2.GaussianBlur(gray,(3,3),0)
# #     kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
# #     kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
# #     img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
# #     img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
# #     edges = img_prewittx + img_prewitty
#     lines = cv2.HoughLinesP(edges,rho=1,theta=np.pi/180, threshold=threshold, minLineLength = minLine, maxLineGap = gap)


    

#     fs = []
#     if lines is None:
#         edge_length1 = 0
#     else:
#         for l in lines:
#             for x1,y1,x2,y2 in l:
#                 fs.append(lengthl(x1,y1,x2,y2))
#         ls = [i/p for i in fs] #normalize to image length
#         lhist = np.histogram(ls, bins = 7, range=(0,1))
#         edge_length1 = lhist[0].tolist()[0] + lhist[0].tolist()[1] + lhist[0].tolist()[2]

    
    
    '''Edges - opencv method'''
    
    '''

    edges = cv2.Canny(img, 100, 200)
    threshold = 100
    labeled, nr_objects = ndimage.label(edges > threshold) 
    unique, lengths = np.unique(labeled, return_counts=True)
    y_e, x_e, g = plt.hist(lengths[1:], bins = 7)
    
    

    
    
    #1st way
    y_min_e = np.where(y_e == y_e.min())[0][0]
    x_min_e = np.mean((x_e[y_min_e:y_min_e + 2]))
    edge_length1 = ((x_max_e * y_e.max()) + (x_min_e * y_e.min())) / (y_e.max() + y_e.min())           # edge length
    
    #2nd way
    y_max_e = np.where(y_e == y_e.max())[0][0]
    x_max_e = np.mean((x_e[y_max_e:y_max_e + 2]))
    edge_length1 = x_max_e
    
    #3rd way
    #edge_length1 = y_e[0] + y_e[1] + y_e[2]
    '''



    
    '''Hue - simplecv method - newly added'''
    img = cv2.imread(filename, 3)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS )

    hue, lig, sat = cv2.split(hls)

    hue = hue.flatten()
    hist = np.histogram(hue, bins = 7, normed=True, range=(0,255))
    hue1 = hist[0].tolist()[0]
    
    
    '''Hue - opencv method'''
    
    '''
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV )
    hue, sat, val = cv2.split(hsv)
    hue = hue.flatten()
    y_h, x_h, bars = plt.hist(hue, bins = 7)
    
    #1st way 
    y_min_h = np.where(y_h == y_h.min())[0][0]
    x_min_h = np.mean((x_h[y_min_h:y_min_h + 2]))
    y_max_h = np.where(y_h == y_h.max())[0][0]
    x_max_h = np.mean((x_h[y_max_h:y_max_h + 2]))
    hue1 = ((x_max_h * y_h.max()) + (x_min_h * y_h.min())) / (y_h.max() + y_h.min())
    
    
    #2nd way
    
    y_max_h = np.where(y_h == y_h.max())[0][0]
    x_max_h = np.mean((x_h[y_max_h:y_max_h + 2]))
    hue1 = x_max_h
    

    
    #3rd way
    #hue1 = y_h[0] + y_h[6] 
    '''





    '''Contrast'''
    #1st method
    imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    contrast = imgGrey.std()                             
    
    
       
    #2nd method - newly added
    # separate channels
    '''
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L,A,B=cv2.split(lab)
    # compute minimum and maximum in 5x5 region using erode and dilate
    kernel = np.ones((5,5),np.uint8)
    min = cv2.erode(L,kernel,iterations = 1)
    max = cv2.dilate(L,kernel,iterations = 1)
    # convert min and max to floats
    min = min.astype(np.float64) 
    max = max.astype(np.float64) 
    # compute local contrast
    contrast = (max-min)/(max+min)
    # get average across whole image
    average_contrast = 100*np.mean(contrast)
    contrast = average_contrast
    '''
    
    
    
    
    '''KP surf'''
    #1st method
#     orb = cv2.ORB_create(nfeatures = 10000) 
#     keypoints, descriptors = orb.detectAndCompute(img, None)
#     kp_surf = len(keypoints)                                          # number of keypoints
    
    
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     surf = cv2.xfeatures2d.SURF_create() 
#     keypoints_surf = surf.detect(img_gray, None)
#     kp_surf = len(keypoints_surf)   
#     #2nd method - newly added
    '''
    orb = cv2.ORB_create(nfeatures = 10000) 
    keypoints = orb.detect(img, None)
    kp_surf = len(keypoints)
    '''


    return [kp_surf, average_perceived_brightness, contrast, area_by_perim, aspect_ratio, edge_length1, hue1]    # returning the values

