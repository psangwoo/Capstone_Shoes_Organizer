
import cv2


def organize(img, background, shoe, pos):
    x, y, x1, y1 = shoe
    shoe_img = img[y:y1, x:x1].copy()
    back_img = background[y:y1, x:x1].copy()
    img[y:y1, x:x1] = back_img
    print(pos[1], pos[1]+y1-y, pos[0], pos[0]+x1-x)
    img[pos[1]:pos[1]+y1-y, pos[0]:pos[0]+x1-x] = shoe_img
    return img
