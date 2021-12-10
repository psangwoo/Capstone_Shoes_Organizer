import cv2, numpy as np
import glob

def __match_shoes(img, gray, to_match):
    accuracy = np.zeros(len(img), dtype='float');

    find_match = np.array([-1 for i in range(len(img))])

    for i in range (0, len(img)):
        accuracy = np.zeros(len(img), dtype='float');
        if not i in to_match:
            continue
        img[i] = cv2.flip(img[i], 1)
        
        for k in range (0, len(img)):
            if not k in to_match:
                continue
            if i == k:
                continue
            #detector = cv2.xfeatures2d.SURF_create()
            detector = cv2.ORB_create()
            
            kp1, desc1 = detector.detectAndCompute(gray[i], None)
            kp2, desc2 = detector.detectAndCompute(gray[k], None)
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            #matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            matches = matcher.match(desc1, desc2)
            
            matches = sorted(matches, key=lambda x:x.distance)
            
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ])
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ])
            
            #print(i, k)
            #print(src_pts, dst_pts)
            
            mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            h,w = img[i].shape[:2]
            pts = np.float32([ [[0,0]],[[0,h-1]],[[w-1,h-1]],[[w-1,0]] ])
            dst = cv2.perspectiveTransform(pts,mtrx)
            accuracy[k]= np.float(float(mask.sum()) / mask.size * 100.0)
            
        img[i] = cv2.flip(img[i], 1)
        #print(accuracy)
        if accuracy.max() != 0:
            print("%d th shoe is pair with %d th shoe, with accuracy %.2f%%" % (i + 1, accuracy.argmax() + 1, accuracy.max()))
            find_match[i] = accuracy.argmax();
    return find_match

def match_shoes(image, boxes):
    img = [image[y:y1, x:x1] for x,y,x1,y1 in boxes]
    
    """
    for i, im in enumerate(img):
        cv2.imshow(str(i), im)
        cv2.waitKey()
    """
    gray = []
    
    for im in img:
        gray.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    
    to_match = [i for i in range(len(boxes))]
    
    return_match = np.zeros(len(boxes))
    while len(to_match)!=0:
        find_match = __match_shoes(img, gray, to_match)
        for i in range (0, len(boxes)):
            if int(find_match[i]) != -1 and find_match[int(find_match[i])] == i:
                return_match[i] = int(find_match[i])
                return_match[int(find_match[i])] = i
                to_match.remove(i)
                print("find_match",int(find_match[i]),i)

    match = []
    for i in range(0, len(boxes)):
        match.append(set([i, int(return_match[i])]))
    print(find_match)
    result = [] # 중복 제거된 값들이 들어갈 리스트

    for value in match:
        if value not in result:
            result.append(value)
    for i in range(len(result)):
        result[i] = list(result[i])
    print(result)
    return result
