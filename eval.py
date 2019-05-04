from keras.applications.resnet50 import preprocess_input
import numpy as np
import os
import cv2

from model import RFBText
import east_utils
import re


def get_path_list(root_path):
    filename_list = list()
    for root, _, filenames in os.walk(root_path):
        filename_list.extend([(os.path.join(root, filename), filename) for filename in filenames
                              if filename.lower().endswith(('jpeg', 'png', 'bmp', 'jpg'))
                             ])

    return filename_list


def paint_boxes_to_image(image, predicted_boxes):
    """
    Draw the quad-boxes on-to the image
    :param image:
    :param predicted_boxes:
    :return:
    """
    for box in predicted_boxes:
        cv2.polylines(image,
                      [box.astype(np.int32).reshape((-1, 1, 2))],
                      True,
                      color=(0, 0, 255),
                      thickness=2)

    return image


def predict_single_image(model, image):
    """
    Predict the bounding box quads for a single image
    :param model:
    :param image:
    :return:
    """
    im_resized, (ratio_h, ratio_w) = east_utils.resize_image(image, max_side_len=2400)
    im_resized = cv2.cvtColor(im_resized, cv2.COLOR_BGR2RGB)

    image_batch = np.expand_dims(im_resized, axis=0)

    image_batch = preprocess_input(image_batch)

    preds = model.predict(image_batch)
    score, geometry = preds

    boxes = east_utils.detect(score_map=score,
                              geo_map=geometry,
                              score_map_thresh=0.8,
                              box_thresh=0.1,
                              nms_thres=0.2)

    predicted_boxes = list()
    if boxes is not None:
        boxes = boxes[:, :8].reshape((-1, 4, 2))
        boxes[:, :, 0] /= ratio_w
        boxes[:, :, 1] /= ratio_h

        for box in boxes:
            box = east_utils.sort_poly(box.astype(np.int32))

            # reduce false positives
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm( box[3] - box[0]) < 5:
                continue

            predicted_boxes.append(box)

    return predicted_boxes


def evaluate_image(model, test_images_path):
    print('predicting from test_images at path: {0}'.format(test_images_path))

    input_path = os.path.join(test_images_path, 'evaluation_images')
    pred_path = os.path.join(test_images_path, 'predicted_boxes')

    if not os.path.isdir(test_images_path):
        os.makedirs(test_images_path)
        os.makedirs(input_path)    
        os.makedirs(pred_path)    

    elif not os.path.isdir(input_path):
        os.makedirs(input_path)    
    
    elif not os.path.isdir(pred_path):
        os.makedirs(pred_path) 

    input_path_list = get_path_list(input_path)

    image_number_pattern = re.compile(r'img_(\d*).jpg')

    for image_path, filename in input_path_list:
        image = cv2.imread(image_path)

        image_copy = np.copy(image)

        print('prediction in progress')
        predicted_boxes = predict_single_image(model, image_copy)
        print('prediction done')

        matcher = image_number_pattern.match(filename)
        image_number = matcher.group(1)

        txt_path = os.path.join(pred_path, 'res_img_' + image_number + '.txt')

        with open(txt_path, 'w') as txt_file:
            for box in predicted_boxes:
                txt_file.write('{},{},{},{},{},{},{},{}\r\n'.format(
                                box[0][0],
                                box[0][1],
                                box[1][0],
                                box[1][1],
                                box[2][0],
                                box[2][1],
                                box[3][0],
                                box[3][1]))


if __name__ == '__main__':
    CWD = '.'

    model_path = os.path.join(CWD, 'model', 'epoch_258_loss_0.010577003471553326.hdf5')

    rfb_text_model = RFBText(model_weights_path=model_path)()
    evaluate_image(model=rfb_text_model,
                   test_images_path=os.path.join(CWD, 'eval_images'))

