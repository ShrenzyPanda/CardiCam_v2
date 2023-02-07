import easyocr
import cv2
import numpy as np


def count_greenary(img, bbox):
    x1,y1,x2,y2 = bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1]
    img_hsv = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)    
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])
    mask = cv2.inRange(img_hsv, lower_green, upper_green)
    return np.count_nonzero(mask)


class ScreenOCR(object):
    def __init__(self):
        self.reader = easyocr.Reader(['en'])
    
    def read_vitals(self, image, get_mAP=True, conf_thres=.4):
        results = self.reader.readtext(image)
        nums, mAPs, sys_sl_dia, sys_sp_dia, others = [], [], [], [], []
        for result in results:
            bbox, reading, conf = result
            if conf < conf_thres: continue
            try:
                num = float(reading)
                nums.append((num, bbox, conf))
            except ValueError:
                if get_mAP:
                    mAP_match = re.search(r'^\(|\)$', reading)
                if get_mAP and mAP_match:
                    mAPs.append((reading, bbox, conf))
                elif reading.count('/')==1:
                    sys_sl_dia.append((reading, bbox, conf))
                else:
                    others.append((reading, bbox, conf))
        nums.sort(key=lambda x: (x[1][2][0]-x[1][0][0]) * (x[1][3][1]-x[1][1][1]), reverse=True)
        return (nums, mAPs, sys_dia, others)


    def sanity_check_values(self, groups):
        """
        SpO2 -> between 75 - 100
        HR -> between 60 - 150
        mAP -> avg of sys and dia and between 80 - 120
        Sys -> 
        """
        pass

    def match_vitals_by_logic(self, groups):
        """
        RR -> in nums nearest to 20 (limit by 0-40)
        HR -> in nums the most green one and largest area
        SpO2 -> in nums nearest to 100 and closest to bbox with text 'sp*2' or 'sp*z'
        mAP -> in mAPs digits between '()'
        Sys -> 2-3 digits before '/'
        Dia -> 2-3 digits after '/'
        """
        pass

    def match_vitals_by_position(self, groups, screen_type):
        pass