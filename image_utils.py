import pandas as pd
import cv2
from sklearn.preprocessing import binarize

def get_img_shape(path):
    """
    Returns HEIGHT, WIDTH, CHANNELS 
    """
    img = cv2.imread(path)
    try:
        return img.shape
    except AttributeError:
        print('error! ', path)
        return (None, None, None)

def superimpose_two_masks(mask_fn1, mask_fn2):
    img_in = cv2.imread(mask_fn1, cv2.IMREAD_GRAYSCALE)
    img_in = binarize(img_in, threshold=50, copy=True)
    img_side = cv2.imread(mask_fn2, cv2.IMREAD_GRAYSCALE)
    img_side = binarize(img_side, threshold=50, copy=True)
    composite = cv2.bitwise_or(img_in,img_side)
    return composite


def get_abs_bbox_loc_df(gdf, WIDTH, HEIGHT):
    """
    :param gdf: given label data frame in normalized format
    :param WIDTH:
    :param HEIGHT:
    :return:
    """
    gdf.columns = ['Class', 'x_center_nml', 'y_center_nml', 'rel_width', 'rel_height' ]
    gdf['x_center'] = gdf['x_center_nml'] *  WIDTH
    gdf['y_center'] = gdf['y_center_nml'] *  HEIGHT
    gdf['width']    = gdf['rel_width'] *  WIDTH
    gdf['width'] = gdf['width'].astype(int)
    gdf['height']   = gdf['rel_height'] *  HEIGHT
    gdf['height'] = gdf['height'].astype(int)
    gdf['x1'] = gdf['x_center'] - gdf['width']/2.
    gdf['x1'] = gdf['x1'].astype(int)
    gdf['y1'] = gdf['y_center'] - gdf['height']/2.
    gdf['y1'] = gdf['y1'].astype(int)
    gdf['x2'] = gdf['x_center'] + gdf['width']/2.
    gdf['x2'] = gdf['x2'].astype(int)
    gdf['y2'] = gdf['y_center'] + gdf['height']/2.
    gdf['y2'] = gdf['y2'].astype(int)
    gdf = gdf[['Class', 'x1', 'y1', 'width', 'height', 'x2', 'y2']]
    gdf = gdf[['Class', 'x1', 'y1','x2', 'y2']]
    return gdf


COLORS = [[  0,  0, 255], [255, 25, 150]] #Fix the color
def draw_bounding_box_ground_truth(img, class_id, x, y, w, h):
    class_id, x, y, w, h = int(class_id), int(x), int(y), int(w), int(h)
    label = str(classes[int(class_id)])
    color = COLORS[int(class_id)]
    cv2.rectangle(img, (x,y), (w,h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

def draw_bounding_box_on_img(img, class_id, x, y, w, h):
    """
    draw bounding box on image
    :param img: image array
    :param class_id:  int id
    :param x: x value top left absolute value in pixel
    :param y: y value top left absolute value in pixel
    :param w: width in pixel
    :param h: heigjt in pixel
    :return: image array
    """
    class_id, x, y, w, h = class_id, int(x), int(y), int(w), int(h)
    label = class_id
    cv2.rectangle(img, (x,y), (x+w,y+h), [255, 25, 150], 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 25, 150], 2)
    return img

def draw_bounding_box_on_img_absolute(img, class_id, x, y, w, h, xywh = True):
    """
    draw bounding box on image
    :param img: image array
    :param class_id:  int id
    :param x: x value top left absolute value in pixel
    :param y: y value top left absolute value in pixel
    :param w: width in pixel
    :param h: heigjt in pixel
    :return: image array
    """
    class_id, x, y, w, h = class_id, int(x), int(y), int(w), int(h)
    if xywh:
        cv2.rectangle(img, (x,y), (x+w,y+h), [255, 25, 150], 2)
    else:
        cv2.rectangle(img, (x,y), (w,h), [255, 25, 150], 2)
    cv2.putText(img, str(class_id), (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 25, 150], 2)
    return img

def bbox_convert_xywh_to_normalized_xywh(bb_df, WIDTH, HEIGHT):
    bb_df.x = (bb_df.x + bb_df.w/2)/WIDTH
    bb_df.y = (bb_df.y + bb_df.h/2)/HEIGHT
    bb_df.w = bb_df.w/WIDTH
    bb_df.h = bb_df.h/HEIGHT
    return bb_df


def plot_the_image_with_bbox(img, gdf):
    """

    :param img:
    :param gdf:
    :return:
    """
    for l in gdf.values:
        img = draw_bounding_box_on_img_absolute(*tuple([img] + list(l)), xywh = False)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'file', (10,100), font, 2, (0, 255, 0), 2, cv2.LINE_AA)
    return img
