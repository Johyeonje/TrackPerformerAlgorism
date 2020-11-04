import os

# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet


class Allocate(object):
    def __init__(self):
        self.index_stack = []   # index used in past
        self.centerX = None        # object center
        self.centerY = None
        self.is_used = 0        # check index used

    def is_exist(self, track_id):
        if track_id in self.index_stack:
            if track_id != self.index_stack[0]:
                print('index matched : ', track_id, '->', self.index_stack[0], self.index_stack)
            return True
        else:
            return False


class Apply_Models(object):
    def __init__(self):
        # Definition of the parameters
        max_cosine_distance = 0.4
        nn_budget = None
        model_filename = 'model_data/mars-small128.pb'
        weights = './checkpoints/yolov4-416'
        self.people_num = 4

        # initialize deep sort
        self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)

        # Set Tracker
        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(self.metric)

        # Load Model
        self.saved_model_loaded = tf.saved_model.load(weights, tags=[tag_constants.SERVING])
        self.infer = self.saved_model_loaded.signatures['serving_default']

        # Create Object Matching Data

        self.person1 = Allocate()
        self.person2 = Allocate()
        self.person3 = Allocate()
        self.person4 = Allocate()

        self.person1.index_stack.append(1)
        self.person2.index_stack.append(2)
        self.person3.index_stack.append(3)
        self.person4.index_stack.append(4)

    def getCenter(self, bbox):
        centerX = (bbox[0] + bbox[2]) / 2
        centerY = (bbox[1] + bbox[3]) / 2

        return centerX, centerY

    def draw_box(self, frame_data, track_id, colors, bbox, class_name='person'):
        color = colors[int(track_id) * 8 % len(colors)]
        color = [j * 255 for j in color]
        cv2.rectangle(frame_data, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        cv2.rectangle(frame_data, (int(bbox[0]), int(bbox[1] - 30)),
                      (int(bbox[0]) + (len(class_name) + len(str(track_id))) * 17, int(bbox[1])),
                      color, -1)
        cv2.putText(frame_data, class_name + "-" + str(track_id),
                    (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75,
                    (255, 255, 255), 2)

    def main(self, frame_data):
        # Definition of the parameters
        nms_max_overlap = 1.0

        # set HyperParams
        size = 416
        iou = 0.45
        score = 0.50
        info = False

        input_size = size

        self.person1.is_used = 0
        self.person2.is_used = 0
        self.person3.is_used = 0
        self.person4.is_used = 0

        out = None

        frame_data = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
        image_data = cv2.resize(frame_data, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        # start_time = time.time()

        batch_data = tf.constant(image_data)
        pred_bbox = self.infer(batch_data)  # Yolo 모델 통과시켜서 바운딩 박스 좌표 반환
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]  # 좌표
            pred_conf = value[:, :, 4:]  # 벡터값

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=iou,
            score_threshold=score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame_data.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        # allowed_classes = list(class_names.values())

        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if count:
            cv2.putText(frame_data, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        2,(0, 255, 0), 2)
            # print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = self.encoder(frame_data, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                      zip(bboxes, scores, names, features)]

        # initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # DeepSort Tracking Start

        # Call the tracker
        self.tracker.predict()  # load tracker
        self.tracker.update(detections)

        match_person = 0
        # reset temp for center compare
        temp = []

        # update tracks
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            # draw bbox on screen           # 이거 처리까지 하고 나서 보내야 할 것 같다.
            bbox = track.to_tlbr()
            class_name = track.get_class()

            # Matching index with index_stack
            if self.person1.is_exist(track.track_id):
                self.person1.centerX, self.person1.centerY = self.getCenter(bbox)
                self.draw_box(frame_data, self.person1.index_stack[0], colors, bbox)
                self.person1.is_used = 1
                match_person += 1
            elif self.person2.is_exist(track.track_id):
                self.person2.centerX, self.person2.centerY = self.getCenter(bbox)
                self.draw_box(frame_data, self.person2.index_stack[0], colors, bbox)
                self.person2.is_used = 1
                match_person += 1
            elif self.person3.is_exist(track.track_id):
                self.person3.centerX, self.person3.centerY = self.getCenter(bbox)
                self.draw_box(frame_data, self.person3.index_stack[0], colors, bbox)
                self.person3.is_used = 1
                match_person += 1
            elif self.person4.is_exist(track.track_id):
                self.person4.centerX, self.person4.centerY = self.getCenter(bbox)
                self.draw_box(frame_data, self.person4.index_stack[0], colors, bbox)
                self.person4.is_used = 1
                match_person += 1
            else:
                temp.append([track.track_id, bbox])
                print('found new object!')

        temp = np.array(temp, dtype=object)

        is_only_one = []

        # Missed Person Only 1
        if len(temp) == 1 and match_person == 3:
            print('if len(temp) == 1 and match_person == 3:')
            if self.person1.is_used == 0:
                print('if self.person1.is_used == 0:')
                is_only_one.append(1)
            if self.person2.is_used == 0:
                print('if self.person2.is_used == 0:')
                is_only_one.append(2)
            if self.person3.is_used == 0:
                print('if self.person3.is_used == 0:')
                is_only_one.append(3)
            if self.person4.is_used == 0:
                print('if self.person4.is_used == 0:')
                is_only_one.append(4)

            # Matching index
            if len(is_only_one) == 1:
                print('if len(is_only_one) == 1:')
                if is_only_one[0] == 1:
                    self.person1.centerX, self.person1.centerY = self.getCenter(temp[0][1])
                    self.person1.index_stack.append(temp[0][0])
                    self.draw_box(frame_data, self.person1.index_stack[0], colors, temp[0][1])
                    self.person1.is_used = 1
                    match_person += 1
                elif is_only_one[0] == 2:
                    self.person2.centerX, self.person2.centerY = self.getCenter(temp[0][1])
                    self.person2.index_stack.append(temp[0][0])
                    self.draw_box(frame_data, self.person2.index_stack[0], colors, temp[0][1])
                    self.person2.is_used = 1
                    match_person += 1
                elif is_only_one[0] == 3:
                    self.person3.centerX, self.person3.centerY = self.getCenter(temp[0][1])
                    self.person3.index_stack.append(temp[0][0])
                    self.draw_box(frame_data, self.person3.index_stack[0], colors, temp[0][1])
                    self.person3.is_used = 1
                    match_person += 1
                elif is_only_one[0] == 4:
                    self.person4.centerX, self.person4.centerY = self.getCenter(temp[0][1])
                    self.person4.index_stack.append(temp[0][0])
                    self.draw_box(frame_data, self.person4.index_stack[0], colors, temp[0][1])
                    self.person4.is_used = 1
                    match_person += 1

        # Missed Person Over 2
        if match_person < 3:
            print('if match_person < 3:', match_person)
            for tmp in temp:
                compare_list = []
                print(tmp)
                nmtX, nmtY = self.getCenter(tmp[1])
                print(nmtX)

                # Apply center location Euclidean Distance
                if not self.person1.is_used:
                    gap = np.sqrt(pow(self.person1.centerX-nmtX, 2)+pow(self.person1.centerY-nmtY, 2))
                    compare_list.append([1, tmp[0], gap])
                if not self.person2.is_used:
                    gap = np.sqrt(pow(self.person2.centerX-nmtX, 2)+pow(self.person2.centerY-nmtY, 2))
                    compare_list.append([2, tmp[0], gap])
                if not self.person3.is_used:
                    gap = np.sqrt(pow(self.person3.centerX-nmtX, 2)+pow(self.person3.centerY-nmtY, 2))
                    compare_list.append([3, tmp[0], gap])
                if not self.person4.is_used:
                    gap = np.sqrt(pow(self.person4.centerX-nmtX, 2)+pow(self.person4.centerY-nmtY, 2))
                    compare_list.append([4, tmp[0], gap])

                # select minimum index
                compare_array = np.array(compare_list)
                search_min = np.swapaxes(compare_array, axis1=0, axis2=1)
                min_idx = np.argmin(search_min[-1])
                self.draw_box(frame_data, compare_list[min_idx][0], colors, tmp[1], class_name)

                # Matching minimum index
                if compare_list[min_idx][0] == 1:
                    self.person1.is_used = 1
                    self.person1.index_stack.append(compare_list[min_idx][1])
                    print('switch with XY-', self.person1.index_stack)
                elif compare_list[min_idx][0] == 2:
                    self.person2.is_used = 1
                    self.person2.index_stack.append(compare_list[min_idx][1])
                    print('switch with XY-', self.person2.index_stack)
                elif compare_list[min_idx][0] == 3:
                    self.person3.is_used = 1
                    self.person3.index_stack.append(compare_list[min_idx][1])
                    print('switch with XY-', self.person3.index_stack)
                elif compare_list[min_idx][0] == 4:
                    self.person4.is_used = 1
                    self.person4.index_stack.append(compare_list[min_idx][1])
                    print('switch with XY-', self.person4.index_stack)
                else:
                    print("something problem in matching with center")


        # if enable info flag then print details about each track
        if info:
            print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id),
                                class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        # calculate frames per second of running detections
        # fps = 1.0 / (time.time() - start_time)
        # print("FPS: %.2f" % fps)
        result = cv2.cvtColor(frame_data, cv2.COLOR_RGB2BGR)

        return result