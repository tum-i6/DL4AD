import argparse
from datetime import datetime
import itertools
import json
import os
from collections import OrderedDict
import cv2
import fiftyone as fo
from fiftyone import ViewField as F
import numpy as np
from PIL import Image
from skimage.measure.entropy import  shannon_entropy
import torch
import tqdm
import yaml
import torchvision
import data
from train import load_model
import utils


class KIA2FiftyOne:
    """
    Class used for loading the test split of the synthetic KIA pedestrian detection dataset to the FiftyOne tool in
    order to support a detailed detection performance analysis for object detection models.
    """

    def __init__(self, root_path):
        """
        Instantiates the class for handling the test split of the synthetic 2D detection dataset from the KIA project
        within FiftyOne.

        :param root_path: The string path to the root folder of the synthetic dataset. The folder must contain all the
        tranches. Must be of type string.
        """
        self.root_path = root_path
        self.transforms = data.get_transform(train=False)

        # KIA dataset path postfixes
        self.img_sample_postfix = 'sensor/camera/left/png'
        self.bbox_2D_postfix = 'ground-truth/2d-bounding-box_json'
        self.instance_segm_postfix = 'ground-truth/semantic-instance-segmentation_png'
        self.keypoint_2D_postfix = 'ground-truth/2d-skeletons_json'
        self.depth_postfix = 'ground-truth/depth_exr'
        self.general_frame_postfix = 'ground-truth/general-globally-per-frame-analysis_json'

        # Defines the keypoint names that are being tracked (tranche 6 & 7 introduced new keypoints that are skipped out
        # of consistency reasons)
        self.keypoint_names = [
            'neck', 'shoulder_r', 'elbow_r', 'wrist_r', 'shoulder_l', 'elbow_l', 'wrist_l', 'pelvis', 'hip_r', 'knee_r',
            'heel_r', 'hip_l', 'knee_l', 'heel_l', 'eye_r', 'eye_l', 'toe_r', 'toe_l', 'shoulder_avg'
        ]

        # Checks whether a cached KIA dataset exists
        print('Looking for cached datasets.\n')
        if 'KIA-Pedestrian-Detection' in fo.list_datasets():
            print('Found cached KIA-Pedestrian-Detection dataset.\n')
            # Loads the cached FiftyOne KIA dataset
            self.dataset = fo.load_dataset('KIA-Pedestrian-Detection')
        else:
            print('No cached datasets available. Creating new KIA-Pedestrian-Detection dataset.\n')
            # Loads the raw KIA dataset for the first time into a FiftyOne dataset instance
            self.dataset = self.upload_kia_data()

        print(self.dataset)  # Logs the dataset properties to the console

    def upload_kia_data(self):
        """
        Loads the KIA pedestrian detection data from the root KIA folder and uploads the test samples to a FiftyOne
        dataset instance.

        :return: A FiftyOne dataset instance, which contains all test samples and annotations from the KIA dataset.
        """
        # Creates the FiftyOne KIA dataset and loads the samples into it
        dataset = fo.Dataset('KIA-Pedestrian-Detection')
        dataset.info = {
            'author': 'your name',
            'created': str(datetime.now())[:-7]
        }
        dataset.default_classes = ['human']
        dataset.default_keypoints = self.keypoint_names
        dataset.persistent = True

        tranches = sorted(os.listdir(f'{self.root_path}'))
        # Iterates over all tranches
        for tranche in tqdm.tqdm(tranches, desc='Loading Tranches', position=0):
            sequences = sorted(os.listdir(f'{self.root_path}/{tranche}'))
            # Iterates over all sequences
            for sequence in tqdm.tqdm(sequences, desc='Loading Sequences', position=1, leave=False, postfix=tranche):
                samples = []  # Used for storing the FiftyOne sample instances
                file_names = sorted(os.listdir(f'{self.root_path}/{tranche}/{sequence}/{self.img_sample_postfix}'))
                # Iterates over all files
                for file in tqdm.tqdm(file_names, desc='Loading Files', position=2, leave=False, postfix=sequence):
                    file_name = file[:-4]  # Removes the .png at the end of the file name
                    # Creates a FiftyOne sample instance
                    # Samples that do not belong to the test split or contain annotation errors are returned as None
                    sample = self.create_kia_sample(tranche, sequence, file_name)
                    if sample is not None:
                        samples.append(sample)  # Adds the sample instance to the list
                if samples:
                    # Adds the samples to the dataset
                    dataset.add_samples(samples)
                    dataset.save()
        return dataset

    def create_kia_sample(self, tranche, sequence, file_name):
        """
        Uses the path information like the KIA root path, the tranche name, the sequence name, the file name and the
        standard path postfixes to localize and extract all information about a specific KIA sample and to load it into
        a FiftyOne sample instance.

        :param tranche: String name of the tranche.
        :param sequence: String name of the sequence.
        :param file_name: String name of the file name without any file format information.
        :return: A FiftyOne sample instance containing the path to the sample image, the respective annotations from
        the KIA dataset and various other image statistics. The function returns None if the sample does not belong to
        the test split or if the sample contains annotation errors.
        """
        raw_file_name = '-'.join(file_name.split('-')[2:])  # Removes the 'arb-camera001-' prefix
        # Path to a specific sequence within a tranche
        data_path = f'{self.root_path}/{tranche}/{sequence}'
        # Defines the paths to the dataset files
        img_sample_path = f'{data_path}/{self.img_sample_postfix}/{file_name}.png'
        instance_segm_path = f'{data_path}/{self.instance_segm_postfix}/{file_name}.png'
        bbox_2D_path = f'{data_path}/{self.bbox_2D_postfix}/{file_name}.json'
        keypoint_2D_path = f'{data_path}/{self.keypoint_2D_postfix}/{file_name}.json'
        depth_path = f'{data_path}/{self.depth_postfix}/{file_name}.exr'
        general_frame_json_path = f'{data_path}/{self.general_frame_postfix}/world-{raw_file_name}.json'

        # Skip samples that are missing crucial annotations (these samples will not be used anymore)
        if not os.path.exists(bbox_2D_path) or \
                not os.path.exists(depth_path) or \
                not os.path.exists(instance_segm_path):
            return None

        # Checks whether the sample belongs to the standard test split
        company_name = tranche.split('_')[2]
        tranche = tranche.split('_')[-1]
        sequence = int(sequence.split('_')[3].split('-')[0].lstrip('0'))
        if sequence in data.standard_kia_split['test_set'][company_name][tranche]:
            split = 'KIA_test_split'
        else:
            return None  # Train and validation samples are skipped

        # Loads the general-globally-per-frame-analysis_json annotations, which contain meta-data for tranches 6 & 7
        pedestrian_meta_data = {}
        if tranche == '6' or tranche == '7':
            with open(general_frame_json_path) as frame_json_file:
                frame_labels = json.load(frame_json_file)  # Loads the annotations from the json file
            if tranche == '7':
                # Loads pedestrian meta-data for tranche 7
                for entity in frame_labels['base_context']['entities']:
                    if 'class_id' in entity:
                        if entity['class_id'] == 'human':
                            pedestrian_meta_data[str(entity['instance_id'])] = {
                                'ood': entity['ood_asset'],
                                'mocap': entity['mocap_asset']
                            }

        # Loads the sample image in RGB and grayscale
        sample_rgb = cv2.cvtColor(cv2.imread(img_sample_path), cv2.COLOR_BGR2RGB)
        sample_gray = cv2.cvtColor(sample_rgb, cv2.COLOR_RGB2GRAY)
        # Loads the instance segmentation image in RGB
        instance_segm_rgb = cv2.cvtColor(cv2.imread(instance_segm_path), cv2.COLOR_BGR2RGB)
        # Extracts a list of FiftyOne detection instances for the sample image (pedestrian instance level data)
        detections = self.extract_sample_gt_detections(
            sample_rgb,
            sample_gray,
            instance_segm_rgb,
            bbox_2D_path,
            keypoint_2D_path,
            pedestrian_meta_data
        )

        # Instantiates the FiftyOne sample instance
        sample = fo.Sample(filepath=img_sample_path)
        # Adds the ground truth information as a new field to the sample
        sample['ground_truth'] = fo.Detections(detections=detections)
        # Stores the information about the company name, tranche name, sequence name, file name and split name
        sample['company_name'] = company_name
        sample['tranche'] = tranche
        sample['sequence'] = sequence
        sample['file_name'] = file_name
        sample.tags.append(split)

        # Extracts some meta-data annotations, which are only present for tranches 6 & 7
        if tranche == '6' or tranche == '7':
            # daytime_type: day, medium, low
            sample['daytime_type'] = frame_labels['base_context']['light_source']['elevation']
            # sky_type: clear, low partly clouded, low completely covered
            sample['sky_type'] = frame_labels['base_context']['light_source']['sky']
            # fog_intensity: factor between 0 and 100
            sample['fog_intensity'] = frame_labels['base_context']['additional_fields']['height_fog']['density'] * 100
            # Extracts some annotations, which are only present for tranche 7
            if tranche == '7':
                # sun_visible: True or False
                sample['sun_visible'] = \
                    bool(frame_labels['base_context']['additional_fields']['scene_light']['sun_visible'])
                # vignette_intensity: factor between 0 and 100
                sample['vignette_intensity'] = \
                    frame_labels['base_context']['additional_fields']['vignette']['intensity'] * 100
                # wetness_type: dry, slightly moist, wet with puddles
                sample['wetness_type'] = frame_labels['base_context']['additional_fields']['wetness']['wetness_zwicky']
                # wetness_intensity: factor between 0 and 100
                sample['wetness_intensity'] = \
                    max(frame_labels['base_context']['additional_fields']['wetness']['wetness'], 0) * 100
                # puddles_intensity: factor between 0 and 100
                sample['puddles_intensity'] = \
                    max(frame_labels['base_context']['additional_fields']['wetness']['puddles'], 0) * 100
                if sample['sun_visible']:
                    # Lens flare is only visible if the sun is visible
                    sample['lens_flare_intensity'] = \
                        frame_labels['base_context']['additional_fields']['lens_flares']['intensity'] * 100
                else:
                    sample['lens_flare_intensity'] = 0
        else:
            # Annotations below are known, since they were only introduced in the newer tranches
            sample['fog_intensity'] = 0
            sample['vignette_intensity'] = 0
            sample['wetness_type'] = 'dry'
            sample['wetness_intensity'] = 0
            sample['puddles_intensity'] = 0
            sample['lens_flare_intensity'] = 0

        # Computes some image statistics
        sample['brightness'] = round(float(sample_gray.mean()) / 255 * 100, 3)
        sample['contrast'] = round(float(sample_gray.std()) / 127.5 * 100, 3)
        sample['edge_strength'] = round(float(utils.get_edge_magnitude(sample_gray).mean()) / 255 * 100, 3)

        # Stores the overall amount of pedestrians and the amount of safety relevant pedestrians
        sample['pedestrian_amount'] = len(detections)
        sample['safety_relevant_pedestrian_amount'] = \
            sum([1 for pedes_instance in detections if not pedes_instance['ignore']])

        return sample

    def extract_sample_gt_detections(self, sample_rgb, sample_gray, instance_segm_img, bbox_2D_path, keypoint_2D_path,
                                     pedestrian_meta_data):
        """
        Uses the sample image, the instance segmentation image, the bbox annotation json file, the keypoint annotation
        json file and the pedestrian_meta_data dictionary to extract various information about every pedestrian
        instance. The information is stored within a FiftyOne detection instance, which is stored inside a list
        alongside other detection instances.

        :param sample_rgb: A RGB numpy array of the sample image.
        :param sample_gray: A grayscale numpy array of the sample image
        :param instance_segm_img: A RGB numpy array of the instance segmentation image.
        :param bbox_2D_path: A string path to the json file, which contains the bbox annotations.
        :param keypoint_2D_path: A string path to the json file, which contains the keypoint annotations.
        :param pedestrian_meta_data: A dictionary, which contains the meta-data for each pedestrian instance. Note that
        the pedestrian_meta_data is only extracted for tranche 7, otherwise it is an empty dictionary.
        :return: A list of FiftyOne detection instances, which contain information about every pedestrian instance
        within a given sample image.
        """
        detections = []  # Used for storing all FiftyOne detection instances for a single sample image
        bboxes = []  # Used for storing all bbox coordinates (in corner format)
        with open(bbox_2D_path) as bbox_json_file:
            bbox_labels = json.load(bbox_json_file)  # Loads the annotations from the json file
        if os.path.exists(keypoint_2D_path):
            keypoints = []  # Used for storing all keypoint annotations
            with open(keypoint_2D_path) as keypoint_json_file:
                keypoint_labels = json.load(keypoint_json_file)  # Loads the annotations from the json file
        else:
            keypoints = None

        # Iterates over all annotated instances from the bbox annotation json file
        for obj in bbox_labels:
            # Filters out objects that are not of the desired class
            if bbox_labels[obj]['class_id'] == 'human':
                # Computes the bbox corner coordinates (upper left and lower right)
                x1 = min(sample_rgb.shape[1] - 1, max(0, bbox_labels[obj]['c_x'] - int(bbox_labels[obj]['w'] / 2)))
                y1 = min(sample_rgb.shape[0] - 1, max(0, bbox_labels[obj]['c_y'] - int(bbox_labels[obj]['h'] / 2)))
                x2 = min(sample_rgb.shape[1] - 1, max(0, bbox_labels[obj]['c_x'] + int(bbox_labels[obj]['w'] / 2)))
                y2 = min(sample_rgb.shape[0] - 1, max(0, bbox_labels[obj]['c_y'] + int(bbox_labels[obj]['h'] / 2)))
                if (x2 - x1) * (y2 - y1) == 0 or x1 >= x2 or y1 >= y2:
                    # Skips the instance if the bbox area is zero or the bbox coordinates are inconsistent
                    continue
                if bbox_labels[obj]['instance_pixels'] < 1 or bbox_labels[obj]['occlusion_est'] > 0.99:
                    # Skips the instance if it is not visible at all
                    continue
                bboxes.append([x1, y1, x2, y2])  # Stores the original bbox coordinates
                # Checks whether the bbox coordinates lie on the edge of the image -> pedestrian is truncated
                if x1 == 0 or x1 == sample_rgb.shape[1]-1 or \
                        y1 == 0 or y1 == sample_rgb.shape[0]-1 or \
                        x2 == 0 or x2 == sample_rgb.shape[1]-1 or \
                        y2 == 0 or y2 == sample_rgb.shape[0]-1:
                    truncated = True
                else:
                    truncated = False
                instance_id = obj
                label = 'human'
                # Normalizes the bbox coordinates (as required by FiftyOne)
                bounding_box = [
                    x1 / sample_rgb.shape[1],
                    y1 / sample_rgb.shape[0],
                    (x2 - x1) / sample_rgb.shape[1],
                    (y2 - y1) / sample_rgb.shape[0]
                ]
                # Extracts some bbox statistics
                bbox_height = int(y2 - y1)
                bbox_width = int(x2 - x1)
                bbox_aspect_ratio = round(bbox_width / bbox_height, 3)
                bbox_area = int(bbox_width * bbox_height)
                visible_instance_pixels = int(bbox_labels[obj]['instance_pixels'])
                foreground_to_bbox_ratio = round(visible_instance_pixels / bbox_area, 3)
                occlusion_ratio = float(bbox_labels[obj]['occlusion_est'])
                if 'occlusion_approx' in bbox_labels[obj]:
                    occlusion_approx = True
                else:
                    occlusion_approx = False
                distance = float(bbox_labels[obj]['distance'])
                ignore = distance > 50  # Ignore instances with a distance > 50m
                if pedestrian_meta_data != {}:
                    mocap = bool(pedestrian_meta_data[instance_id]['mocap'])  # Pedestrian asset with motion capture
                    ood = bool(pedestrian_meta_data[instance_id]['ood'])  # Out of distribution pedestrian asset
                else:
                    mocap = None
                    ood = None

                # Extracts the instance segmentation mask
                # Converts the instance id into a hexadecimal color code
                hex_color_encoding = hex(int(obj)).replace('0x', '').zfill(6)
                # Converts the hexadecimal color code into an RGB color code
                rgb_color = np.array([int(hex_color_encoding[i:i + 2], 16) for i in (0, 2, 4)])
                # Cuts out the bbox from the instance segmentation mask
                mask = instance_segm_img[y1:y2, x1:x2]
                # Produces a binary mask for the given RGB color code
                mask = cv2.inRange(mask, rgb_color, rgb_color) / 255
                # Skips the instance if the instance segmentation mask does not exist
                if not np.any(mask.astype(np.uint8)):
                    # Removes the previously added bbox coordinates
                    bboxes.pop()
                    continue

                # Extracts the keypoint annotations, if they exist
                if os.path.exists(keypoint_2D_path):
                    for keypoint in self.keypoint_names:
                        # Tranche 6 & 7 introduced missing keypoints to signalize their visibility
                        if keypoint in keypoint_labels[obj]:
                            # Image coordinates of a specific keypoint
                            x = keypoint_labels[obj][keypoint][0]
                            y = keypoint_labels[obj][keypoint][1]
                            # Checks whether the keypoint coordinates are inside the bbox
                            if x1 <= x < x2 and y1 <= y < y2:
                                # Checks whether the keypoint coordinates lie inside the instance segmentation mask
                                if mask[int(y-y1), int(x-x1)]:
                                    # Adds the keypoint coordinates to the list
                                    keypoints.append((x, y))
                                    continue
                        # Adds an empty tuple to the list, since the keypoint coordinates are not visible
                        keypoints.append(())

                # Extracts some image statistics w.r. to the bbox
                bbox_gray = sample_gray[y1:y2, x1:x2]
                bbox_brightness = round(float(bbox_gray.mean()), 3) / 255 * 100
                bbox_contrast = round(float(bbox_gray.std()), 3) / 127.5 * 100
                foreground_brightness = round(float(bbox_gray[np.nonzero(mask)].mean()), 3) / 255 * 100
                foreground_contrast = round(float(bbox_gray[np.nonzero(mask)].std()), 3) / 127.5 * 100
                background_mask = 1 - mask
                if np.any(background_mask):
                    background_brightness = round(float(bbox_gray[np.nonzero(background_mask)].mean()), 3) / 255 * 100
                    background_contrast = round(float(bbox_gray[np.nonzero(background_mask)].std()), 3) / 127.5 * 100
                    contrast_to_background = round(abs(background_contrast - foreground_contrast), 3)
                else:
                    background_brightness = None
                    background_contrast = None
                    contrast_to_background = None

                # Measures the Shannon entropy within the bbox area
                entropy = shannon_entropy(bbox_gray)

                # Quantifies the edge strength
                # Applies the dilation and erosion operation to create two new masks
                kernel = np.ones((2, 2), np.uint8)
                dilated_mask = cv2.dilate(mask, kernel, iterations=2)  # larger than original mask
                eroded_mask = cv2.erode(mask, kernel, iterations=2)  # smaller than original mask
                boundary_mask = dilated_mask - eroded_mask
                # Calculates the edge magnitude for the bbox
                magnitude = utils.get_edge_magnitude(bbox_gray)
                # Masks out the magnitude using the dilated mask
                foreground_magnitude = magnitude[np.nonzero(dilated_mask)]
                # Masks out the magnitude using the inverted dilated mask
                background_magnitude = magnitude[np.nonzero(1 - dilated_mask)]
                # Masks out the magnitude using the boundary mask
                boundary_magnitude = magnitude[np.nonzero(boundary_mask)]
                # Calculates the edge strength for the foreground, background and boundary by taking the mean
                foreground_edge_strength = round(float(foreground_magnitude.mean()), 3)
                if np.any(background_magnitude):
                    background_edge_strength = round(float(background_magnitude.mean()), 3)
                else:
                    background_edge_strength = None
                if np.any(boundary_magnitude):
                    boundary_edge_strength = round(float(boundary_magnitude.mean()), 3)
                else:
                    boundary_edge_strength = None

                # Creates a FiftyOne detection instance and adds it to the list
                detections.append(
                    fo.Detection(
                        instance_id=instance_id,
                        label=label,
                        bounding_box=bounding_box,
                        mask=mask,
                        bbox_height=bbox_height,
                        bbox_width=bbox_width,
                        bbox_aspect_ratio=bbox_aspect_ratio,
                        bbox_area=bbox_area,
                        visible_instance_pixels=visible_instance_pixels,
                        foreground_to_bbox_ratio=foreground_to_bbox_ratio,
                        occlusion_ratio=occlusion_ratio,
                        occlusion_approx=occlusion_approx,
                        truncated=truncated,
                        distance=distance,
                        bbox_brightness=bbox_brightness,
                        bbox_contrast=bbox_contrast,
                        foreground_brightness=foreground_brightness,
                        foreground_contrast=foreground_contrast,
                        background_brightness=background_brightness,
                        background_contrast=background_contrast,
                        contrast_to_background=contrast_to_background,
                        entropy=entropy,
                        foreground_edge_strength=foreground_edge_strength,
                        background_edge_strength=background_edge_strength,
                        boundary_edge_strength=boundary_edge_strength,
                        crowdedness=0,
                        motion_capture=mocap,
                        out_of_distribution=ood,
                        ignore=ignore,
                        keypoints=keypoints
                    )
                )

        # Iterates over all possible pairs of bboxes and computes the overlap for each of them
        for box_pair in list(itertools.combinations(range(len(bboxes)), 2)):
            # Computes the corner coordinates for the intersection of the two bboxes
            x1 = max(bboxes[box_pair[0]][0], bboxes[box_pair[1]][0])
            y1 = max(bboxes[box_pair[0]][1], bboxes[box_pair[1]][1])
            x2 = min(bboxes[box_pair[0]][2], bboxes[box_pair[1]][2])
            y2 = min(bboxes[box_pair[0]][3], bboxes[box_pair[1]][3])
            # Computes the area for the intersection
            inter_area = max(x2 - x1, 0) * max(y2 - y1, 0)
            if inter_area != 0:  # No intersection -> overlap is 0
                # Computes the overlap
                overlap1 = inter_area / detections[box_pair[0]]['bbox_area']
                overlap2 = inter_area / detections[box_pair[1]]['bbox_area']
                # Updates the ignore list, if one of the two bboxes is being overlapped by 60% or more (heavily crowded)
                if overlap1 >= 0.6 or overlap2 >= 0.6:
                    if detections[box_pair[0]]['distance'] > detections[box_pair[1]]['distance']:
                        detections[box_pair[0]]['ignore'] = True
                    else:
                        detections[box_pair[1]]['ignore'] = True
            else:
                overlap1 = 0
                overlap2 = 0

            # Uses the overlap values and scales them by the ratio between the two bboxes to compute the crowdedness
            bboxes_ratio = min(detections[box_pair[0]]['bbox_area'], detections[box_pair[1]]['bbox_area']) / \
                           max(detections[box_pair[0]]['bbox_area'], detections[box_pair[1]]['bbox_area'])
            detections[box_pair[0]]['crowdedness'] += overlap1 * bboxes_ratio
            detections[box_pair[1]]['crowdedness'] += overlap2 * bboxes_ratio

        return detections

    def upload_model_predictions(self, checkpoint_path, device_id, cfg):
        """
        Iterates over all samples from the FiftyOne KIA-Pedestrian-Detection dataset and infers them into the given
        detection model, which is loaded from the checkpoint path. The model predictions are then stored within the
        same FiftyOne sample instances.

        :param checkpoint_path: String path to a checkpoint file, which contains the model weights.
        :param device_id: The integer ID of the GPU, which should be used for inference.
        """
        #model_name = checkpoint_path.split('/')[-2].split('_')[-1]  # Extracts the model name from the checkpoint path
        model_name = cfg['model_name']  # Extracts the model name from the checkpoint path
        detection_task = model_name.split('-')[-1]

        # Loads the detection model and the weights from the checkpoint file
        checkpoint = torch.load(checkpoint_path)
        model, device = load_model(cfg)
        model.load_state_dict(checkpoint['model'])
        model.eval()

        # Iterates over all dataset samples and infers them into the detection model to generate the predictions
        with fo.ProgressBar() as pb:
            for sample in pb(self.dataset):
                # Loads the sample image
                img = Image.open(sample.filepath)
                img, _ = self.transforms(img, None)
                c, h, w = img.shape

                # Extracts the predictions from the model for the given sample
                preds = model([img.to(device)])[0]
                labels = preds['labels'].cpu().detach().numpy()
                scores = preds['scores'].cpu().detach().numpy()
                boxes = preds['boxes'].cpu().detach().numpy()
                if 'IS' in detection_task:
                    masks = preds['masks'].cpu().detach().numpy()
                if 'KD' in detection_task:
                    keypoints = preds['keypoints'].cpu().detach().numpy()

                detections = []  # Used for storing all FiftyOne detection instances for a single sample
                for i, label, score, box in zip(range(len(labels)), labels, scores, boxes):
                    # Extracts the predicted bbox coordinates
                    x1, y1, x2, y2 = box
                    rel_box = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]

                    # Stores the predictions for the 'OD' detection task inside a FiftyOne detection instance
                    detection = fo.Detection(
                        label='human',  # There is only one class in KIA ('human')
                        bounding_box=rel_box,
                        confidence=score
                    )

                    if 'IS' in detection_task:
                        # Stores the predicted instance segmentation mask with threshold 0.5 ('IS' task)
                        detection['mask'] = masks[i][0, int(y1):int(y2), int(x1):int(x2)] >= 0.5

                    if 'KD' in detection_task:
                        keypoints_list = []  # Stores the predicted keypoints
                        confidences = []  # Stores the corresponding confidence scores for each keypoint
                        for keypoint in keypoints[i]:
                            keypoints_list.append((int(keypoint[0]), int(keypoint[1])))
                            confidences.append(float(keypoint[2]))
                        # Stores the predictions for the 'KD' task
                        detection['keypoints'] = keypoints_list
                        detection['keypoint_scores'] = confidences

                    # Stores the predictions as a FiftyOne detection instance inside the detections list
                    detections.append(detection)

                # Adds the extracted model predictions to the original FiftyOne sample instance
                sample[f'prediction_{model_name}'] = fo.Detections(detections=detections)
                sample.save()

        # Runs the evaluation for the model predictions
        self.evaluate_model_predictions(f'prediction_{model_name}', detection_task, checkpoint['conf_thr'])
    def upload_model_predictions_heatmaps(self, checkpoint_path, device_id, cfg):
        """
        Iterates over all samples from the FiftyOne KIA-Pedestrian-Detection dataset and infers them into the given
        detection model, which is loaded from the checkpoint path. The model predictions are then stored within the
        same FiftyOne sample instances.

        :param checkpoint_path: String path to a checkpoint file, which contains the model weights.
        :param device_id: The integer ID of the GPU, which should be used for inference.
        """
        #model_name = checkpoint_path.split('/')[-2].split('_')[-1]  # Extracts the model name from the checkpoint path
        model_name = cfg['model_name']  # Extracts the model name from the checkpoint path
        detection_task = model_name.split('-')[-1]

        # Loads the detection model and the weights from the checkpoint file
        checkpoint = torch.load(checkpoint_path)
        model, device = load_model(cfg)
        model.load_state_dict(checkpoint['model'])
        model.eval()

        # register hooks on the feature maps
        feature_maps = []
        model.backbone.register_forward_hook(lambda module, input, output: feature_maps.append(output))
        heatmap_dir=f'{cfg["heatmap_path"]}/{cfg["dataset"]}/{cfg["model_name"]}'
        if not os.path.exists(heatmap_dir):
            os.makedirs(heatmap_dir)
        # Iterates over all dataset samples and infers them into the detection model to generate the predictions

        with fo.ProgressBar() as pb:
            sample_id=0
            processed_sample_list=[]
            for sample in pb(self.dataset):
                if sample_id>=5000:
                    break
                # Loads the sample image
                img = Image.open(sample.filepath)
                img, _ = self.transforms(img, None)
                c, h, w = img.shape

                # Extracts the predictions from the model for the given sample
                preds = model([img.to(device)])[0]
                labels = preds['labels'].cpu().detach().numpy()
                scores = preds['scores'].cpu().detach().numpy()
                boxes = preds['boxes'].cpu().detach().numpy()
                if 'IS' in detection_task:
                    masks = preds['masks'].cpu().detach().numpy()
                if 'KD' in detection_task:
                    keypoints = preds['keypoints'].cpu().detach().numpy()

                detections = []  # Used for storing all FiftyOne detection instances for a single sample
                for i, label, score, box in zip(range(len(labels)), labels, scores, boxes):
                    # Extracts the predicted bbox coordinates
                    x1, y1, x2, y2 = box
                    rel_box = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]

                    # Stores the predictions for the 'OD' detection task inside a FiftyOne detection instance
                    detection = fo.Detection(
                        label='human',  # There is only one class in KIA ('human')
                        bounding_box=rel_box,
                        confidence=score
                    )
                    if 'IS' in detection_task:
                        # Stores the predicted instance segmentation mask with threshold 0.5 ('IS' task)
                        detection['mask'] = masks[i][0, int(y1):int(y2), int(x1):int(x2)] >= 0.5

                    if 'KD' in detection_task:
                        keypoints_list = []  # Stores the predicted keypoints
                        confidences = []  # Stores the corresponding confidence scores for each keypoint
                        for keypoint in keypoints[i]:
                            keypoints_list.append((int(keypoint[0]), int(keypoint[1])))
                            confidences.append(float(keypoint[2]))
                        # Stores the predictions for the 'KD' task
                        detection['keypoints'] = keypoints_list
                        detection['keypoint_scores'] = confidences

                    # Stores the predictions as a FiftyOne detection instance inside the detections list
                    detections.append(detection)


                # Adds the extracted model predictions to the original FiftyOne sample instance
                sample[f'prediction_{model_name}_heatmaps'] = fo.Detections(detections=detections)
                sample.save()
                # ------------------- creat binary mask prediction -------------------
                activations = feature_maps.pop()
                if isinstance(activations, torch.Tensor):
                    activations = OrderedDict([("0", activations)])
                activations = {k: v.detach().cpu() for k, v in activations.items()}

                heatmap_avg = torch.mean(activations['0'], 1)  # (1, H, W)

                resized_heatmap = torchvision.transforms.Resize(size=(h, w), antialias=True)(heatmap_avg)[0]
                heatmap_mask = torch.nn.Sigmoid()(resized_heatmap)
                heatmap_mask = heatmap_mask.cpu().numpy()*100  # (H, W)
                cv2.imwrite(f'{heatmap_dir}/{sample.file_name}.png', heatmap_mask)
                sample_id+=1
                processed_sample_list.append(sample.id)

        # Runs the evaluation for the model predictions
        self.evaluate_model_predictions(f'prediction_{model_name}_heatmaps', detection_task, checkpoint['conf_thr'], processed_sample_list)

    def evaluate_model_predictions(self, field_name, detection_task, conf_thr, processed_sample_list=[]):
        """
        Evaluates the model predictions, which are stored in a specific field within each FiftyOne sample instance.

        :param field_name: The name of the FiftyOne sample field, where the model predictions are stored.
        :param detection_task: String that specifies the detection task (OD or IS).
        :param conf_thr: The model specific confidence threshold for F1-score maximization.
        """
        # Filters out the low confidence detections
        high_conf_view = self.dataset.filter_labels(field_name, F('confidence') >= float(conf_thr), only_matches=False)
        if len(processed_sample_list)>0:
            high_conf_view=high_conf_view.select(processed_sample_list)
        # Runs the evaluation
        high_conf_view.evaluate_detections(
            field_name,
            gt_field='ground_truth',
            eval_key=f'eval_{field_name.replace("-", "_")}',
            iou=0.25,
            use_masks=False,
            tolerance=2
        )


class CityPersons2FiftyOne:
    """
    Class used for loading the validation split of the CityPersons pedestrian detection dataset to the FiftyOne tool in
    order to support a detailed detection performance analysis for object detection models. Note that the original
    CityPersons test split does not contain public labels, therefor the validation dataset is used as test data. From
    here on, the docs will refer to the official validation split as the test split.
    """

    def __init__(self, root_path):
        """
        Instantiates the class for handling the test split of the CityPersons dataset within FiftyOne.

        :param root_path: The string path to the root folder of the dataset. Must be of type string.
        """
        self.root_path = root_path
        self.transforms = data.get_transform(train=False)

        # CityPersons dataset path postfixes
        self.img_sample_postfix = 'leftImg8bit'
        self.bbox_2D_postfix = 'gtBboxCityPersons'
        self.instance_segm_postfix = 'gtFine'
        self.depth_postfix = 'disparity'

        # Checks whether a cached CityPersons dataset exists
        print('Looking for cached datasets.\n')
        if 'CityPersons' in fo.list_datasets():
            print('Found cached CityPersons dataset.\n')
            # Loads the cached FiftyOne CityPersons dataset
            self.dataset = fo.load_dataset('CityPersons')
        else:
            print('No cached datasets available. Creating new CityPersons dataset.\n')
            # Loads the raw CityPersons dataset for the first time into a FiftyOne dataset instance
            self.dataset = self.upload_citypersons_data()

        print(self.dataset)  # Logs the dataset properties to the console

    def upload_citypersons_data(self):
        """
        Loads the CityPersons pedestrian detection data from the root folder and uploads the test samples to a FiftyOne
        dataset instance.

        :return: A FiftyOne dataset instance, which contains all test samples and annotations from the CityPersons
        dataset.
        """
        # Creates the FiftyOne CityPersons dataset and loads the samples into it
        dataset = fo.Dataset('CityPersons')
        dataset.info = {
            'author': 'your name',
            'created': str(datetime.now())[:-7]
        }
        dataset.default_classes = ['human']
        dataset.persistent = True

        city_folders = sorted(os.listdir(f'{self.root_path}/{self.bbox_2D_postfix}/val'))
        for city in tqdm.tqdm(city_folders, desc='Loading Cities', position=0):  # Iterates over all city folders
            file_names = sorted(os.listdir(f'{self.root_path}/{self.bbox_2D_postfix}/val/{city}'))
            samples = []  # Used for storing the FiftyOne sample instances
            for file in tqdm.tqdm(file_names, desc='Loading Files', position=1, leave=False):  # Iterates over all files
                raw_file_name = '_'.join(file.split('_')[:-1])
                # Creates a FiftyOne sample instance
                sample = self.create_citypersons_sample(city, raw_file_name)
                samples.append(sample)  # Adds the sample instance to the list
            # Adds the samples to the dataset
            dataset.add_samples(samples)
            dataset.save()
        return dataset

    def create_citypersons_sample(self, city, raw_file_name):
        """
        Uses the path information like the name of the city, the raw file name and the standard path postfixes to
        localize and extract all information about a specific CityPersons sample and to load it into a FiftyOne sample
        instance.

        :param city: String name of the city.
        :param raw_file_name: String name of the file name without any file format and postfix information.
        :return: A FiftyOne sample instance containing the path to the sample image, the respective annotations from
        the CityPersons dataset and various other image statistics.
        """
        # Defines the paths to the dataset files
        img_sample_path = \
            f'{self.root_path}/{self.img_sample_postfix}/val/{city}/{raw_file_name}_leftImg8bit.png'
        bbox_2D_path = \
            f'{self.root_path}/{self.bbox_2D_postfix}/val/{city}/{raw_file_name}_gtBboxCityPersons.json'
        instance_segm_path = \
            f'{self.root_path}/{self.instance_segm_postfix}/val/{city}/{raw_file_name}_gtFine_instanceIds.png'

        # Loads the sample image in RGB and grayscale
        sample_rgb = cv2.cvtColor(cv2.imread(img_sample_path), cv2.COLOR_BGR2RGB)
        sample_gray = cv2.cvtColor(sample_rgb, cv2.COLOR_RGB2GRAY)
        # Loads the instance segmentation image
        instance_segm = cv2.imread(instance_segm_path, cv2.IMREAD_UNCHANGED)
        # Extracts a list of FiftyOne detection instances for the sample image (pedestrian instance level data)
        detections = self.extract_sample_gt_detections(
            sample_rgb,
            sample_gray,
            instance_segm,
            bbox_2D_path
        )

        # Instantiates the FiftyOne sample instance
        sample = fo.Sample(filepath=img_sample_path)
        # Adds the ground truth information as a new field to the sample
        sample['ground_truth'] = fo.Detections(detections=detections)
        # Stores the information about the name of the city, the file name and the split name
        sample['city'] = city
        sample['file_name'] = raw_file_name
        sample.tags.append('CityPersons_test_split')

        # Computes some image statistics
        sample['brightness'] = round(float(sample_gray.mean()) / 255 * 100, 3)
        sample['contrast'] = round(float(sample_gray.std()) / 127.5 * 100, 3)
        sample['edge_strength'] = round(float(utils.get_edge_magnitude(sample_gray).mean()) / 255 * 100, 3)

        # Stores the overall amount of pedestrians and the amount of safety relevant pedestrians
        sample['pedestrian_amount'] = len(detections)
        sample['safety_relevant_pedestrian_amount'] = \
            sum([1 for pedes_instance in detections if not pedes_instance['ignore']])

        return sample

    def extract_sample_gt_detections(self, sample_rgb, sample_gray, instance_segm_img, bbox_2D_path):
        """
        Uses the sample image, the instance segmentation image and the bbox annotation json file to extract various
        information about every pedestrian instance. The information is stored within a FiftyOne detection instance,
        which is stored inside a list alongside other detection instances.

        :param sample_rgb: A RGB numpy array of the sample image.
        :param sample_gray: A grayscale numpy array of the sample image
        :param instance_segm_img: A numpy array of the instance segmentation image.
        :param bbox_2D_path: A string path to the json file, which contains the bbox annotations.
        :return: A list of FiftyOne detection instances, which contain information about every pedestrian instance
        within a given sample image.
        """
        detections = []  # Used for storing all FiftyOne detection instances for a single sample image
        bboxes = []  # Used for storing all bbox coordinates (in corner format)
        with open(bbox_2D_path) as bbox_json_file:
            bbox_labels = json.load(bbox_json_file)  # Loads the annotations from the json file

        # Iterates over all annotated instances from the bbox annotation json file
        for obj in bbox_labels['objects']:
            # Filters out objects that are not of the desired class
            if obj['label'] != 'ignore':
                # Computes the bbox corner coordinates (upper left and lower right)
                x1 = min(2048 - 1, max(0, obj['bbox'][0]))
                y1 = min(1024 - 1, max(0, obj['bbox'][1]))
                x2 = min(2048 - 1, max(0, obj['bbox'][0] + int(obj['bbox'][2])))
                y2 = min(1024 - 1, max(0, obj['bbox'][1] + int(obj['bbox'][3])))
                if (x2 - x1) * (y2 - y1) == 0 or x1 >= x2 or y1 >= y2:
                    # Skips the instance if the bbox area is zero or the bbox coordinates are inconsistent
                    continue
                if obj['instance_pixels'] < 1 or obj['occlusion_est'] > 0.99:
                    # Skips the instance if it is not visible at all
                    continue
                bboxes.append([x1, y1, x2, y2])  # Stores the original bbox coordinates
                # Checks whether the bbox coordinates lie on the edge of the image -> pedestrian is truncated
                if x1 == 0 or x1 == sample_rgb.shape[1]-1 or \
                        y1 == 0 or y1 == sample_rgb.shape[0]-1 or \
                        x2 == 0 or x2 == sample_rgb.shape[1]-1 or \
                        y2 == 0 or y2 == sample_rgb.shape[0]-1:
                    truncated = True
                else:
                    truncated = False
                instance_id = str(obj['instanceId'])
                label = 'human'
                # Normalizes the bbox coordinates (as required by FiftyOne)
                bounding_box = [
                    x1 / sample_rgb.shape[1],
                    y1 / sample_rgb.shape[0],
                    (x2 - x1) / sample_rgb.shape[1],
                    (y2 - y1) / sample_rgb.shape[0]
                ]
                citypersons_label = obj['label']
                # Extracts some bbox statistics
                bbox_height = int(y2 - y1)
                bbox_width = int(x2 - x1)
                bbox_aspect_ratio = round(bbox_width / bbox_height, 3)
                bbox_area = int(bbox_width * bbox_height)
                visible_instance_pixels = int(obj['instance_pixels'])
                foreground_to_bbox_ratio = round(visible_instance_pixels / bbox_area, 3)
                occlusion_ratio = float(obj['occlusion_est'])
                distance = float(obj['distance'])
                ignore = distance > 50  # Ignore instances with a distance > 50m
                iscrowd = bool(obj['label'] == 'person group')

                # Extracts the instance segmentation mask
                mask = cv2.inRange(instance_segm_img[y1:y2, x1:x2], obj['instanceId'], obj['instanceId']) / 255
                # Skips the instance if the instance segmentation mask does not exist
                if not np.any(mask.astype(np.uint8)):
                    # Removes the previously added bbox coordinates
                    bboxes.pop()
                    continue

                # Extracts some image statistics w.r. to the bbox
                bbox_gray = sample_gray[y1:y2, x1:x2]
                bbox_brightness = round(float(bbox_gray.mean()), 3) / 255 * 100
                bbox_contrast = round(float(bbox_gray.std()), 3) / 127.5 * 100
                foreground_brightness = round(float(bbox_gray[np.nonzero(mask)].mean()), 3) / 255 * 100
                foreground_contrast = round(float(bbox_gray[np.nonzero(mask)].std()), 3) / 127.5 * 100
                background_mask = 1 - mask
                if np.any(background_mask):
                    background_brightness = round(float(bbox_gray[np.nonzero(background_mask)].mean()), 3) / 255 * 100
                    background_contrast = round(float(bbox_gray[np.nonzero(background_mask)].std()), 3) / 127.5 * 100
                    contrast_to_background = round(abs(background_contrast - foreground_contrast), 3)
                else:
                    background_brightness = None
                    background_contrast = None
                    contrast_to_background = None

                # Measures the Shannon entropy within the bbox area
                entropy = shannon_entropy(bbox_gray)

                # Quantifies the edge strength
                # Applies the dilation and erosion operation to create two new masks
                kernel = np.ones((2, 2), np.uint8)
                dilated_mask = cv2.dilate(mask, kernel, iterations=2)  # larger than original mask
                eroded_mask = cv2.erode(mask, kernel, iterations=2)  # smaller than original mask
                boundary_mask = dilated_mask - eroded_mask
                # Calculates the edge magnitude for the bbox
                magnitude = utils.get_edge_magnitude(bbox_gray)
                # Masks out the magnitude using the dilated mask
                foreground_magnitude = magnitude[np.nonzero(dilated_mask)]
                # Masks out the magnitude using the inverted dilated mask
                background_magnitude = magnitude[np.nonzero(1 - dilated_mask)]
                # Masks out the magnitude using the boundary mask
                boundary_magnitude = magnitude[np.nonzero(boundary_mask)]
                # Calculates the edge strength for the foreground, background and boundary by taking the mean
                foreground_edge_strength = round(float(foreground_magnitude.mean()), 3)
                if np.any(background_magnitude):
                    background_edge_strength = round(float(background_magnitude.mean()), 3)
                else:
                    background_edge_strength = None
                if np.any(boundary_magnitude):
                    boundary_edge_strength = round(float(boundary_magnitude.mean()), 3)
                else:
                    boundary_edge_strength = None

                # Creates a FiftyOne detection instance and adds it to the list
                detections.append(
                    fo.Detection(
                        instance_id=instance_id,
                        label=label,
                        bounding_box=bounding_box,
                        mask=mask,
                        citypersons_label=citypersons_label,
                        bbox_height=bbox_height,
                        bbox_width=bbox_width,
                        bbox_aspect_ratio=bbox_aspect_ratio,
                        bbox_area=bbox_area,
                        visible_instance_pixels=visible_instance_pixels,
                        foreground_to_bbox_ratio=foreground_to_bbox_ratio,
                        occlusion_ratio=occlusion_ratio,
                        truncated=truncated,
                        distance=distance,
                        bbox_brightness=bbox_brightness,
                        bbox_contrast=bbox_contrast,
                        foreground_brightness=foreground_brightness,
                        foreground_contrast=foreground_contrast,
                        background_brightness=background_brightness,
                        background_contrast=background_contrast,
                        contrast_to_background=contrast_to_background,
                        entropy=entropy,
                        foreground_edge_strength=foreground_edge_strength,
                        background_edge_strength=background_edge_strength,
                        boundary_edge_strength=boundary_edge_strength,
                        crowdedness=0,
                        ignore=ignore,
                        iscrowd=iscrowd
                    )
                )

        # Iterates over all possible pairs of bboxes and computes the overlap for each of them
        for box_pair in list(itertools.combinations(range(len(bboxes)), 2)):
            # Computes the corner coordinates for the intersection of the two bboxes
            x1 = max(bboxes[box_pair[0]][0], bboxes[box_pair[1]][0])
            y1 = max(bboxes[box_pair[0]][1], bboxes[box_pair[1]][1])
            x2 = min(bboxes[box_pair[0]][2], bboxes[box_pair[1]][2])
            y2 = min(bboxes[box_pair[0]][3], bboxes[box_pair[1]][3])
            # Computes the area for the intersection
            inter_area = max(x2 - x1, 0) * max(y2 - y1, 0)
            if inter_area != 0:  # No intersection -> overlap is 0
                # Computes the overlap
                overlap1 = inter_area / detections[box_pair[0]]['bbox_area']
                overlap2 = inter_area / detections[box_pair[1]]['bbox_area']
                # Updates the ignore list, if one of the two bboxes is being overlapped by 60% or more (heavily crowded)
                if overlap1 >= 0.6 or overlap2 >= 0.6:
                    if detections[box_pair[0]]['distance'] > detections[box_pair[1]]['distance']:
                        detections[box_pair[0]]['ignore'] = True
                    else:
                        detections[box_pair[1]]['ignore'] = True
            else:
                overlap1 = 0
                overlap2 = 0

            # Uses the overlap values and scales them by the ratio between the two bboxes to compute the crowdedness
            bboxes_ratio = min(detections[box_pair[0]]['bbox_area'], detections[box_pair[1]]['bbox_area']) / \
                           max(detections[box_pair[0]]['bbox_area'], detections[box_pair[1]]['bbox_area'])
            detections[box_pair[0]]['crowdedness'] += overlap1 * bboxes_ratio
            detections[box_pair[1]]['crowdedness'] += overlap2 * bboxes_ratio

        return detections

    def upload_model_predictions(self, checkpoint_path, device_id):
        """
        Iterates over all samples from the FiftyOne CityPersons dataset and infers them into the given detection model,
        which is loaded from the checkpoint path. The model predictions are then stored within the same FiftyOne sample
        instances.

        :param checkpoint_path: String path to a checkpoint file, which contains the model weights.
        :param device_id: The integer ID of the GPU, which should be used for inference.
        """
        model_name = checkpoint_path.split('/')[-2].split('_')[-1]  # Extracts the model name from the checkpoint path
        detection_task = model_name.split('-')[-1]

        # Loads the detection model and the weights from the checkpoint file
        checkpoint = torch.load(checkpoint_path)
        model, device = load_model({'model_name': model_name, 'gpu_id': device_id})
        model.load_state_dict(checkpoint['model'])
        model.eval()

        # Iterates over all dataset samples and infers them into the detection model to generate the predictions
        with fo.ProgressBar() as pb:
            for sample in pb(self.dataset):
                # Loads the sample image
                img = Image.open(sample.filepath)
                img, _ = self.transforms(img, None)
                c, h, w = img.shape

                # Extracts the predictions from the model for the given sample
                preds = model([img.to(device)])[0]
                labels = preds['labels'].cpu().detach().numpy()
                scores = preds['scores'].cpu().detach().numpy()
                boxes = preds['boxes'].cpu().detach().numpy()
                if 'IS' in detection_task:
                    masks = preds['masks'].cpu().detach().numpy()

                detections = []  # Used for storing all FiftyOne detection instances for a single sample
                for i, label, score, box in zip(range(len(labels)), labels, scores, boxes):
                    # Extracts the predicted bbox coordinates
                    x1, y1, x2, y2 = box
                    rel_box = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]

                    # Stores the predictions for the 'OD' detection task inside a FiftyOne detection instance
                    detection = fo.Detection(
                        label='human',
                        bounding_box=rel_box,
                        confidence=score
                    )

                    if 'IS' in detection_task:
                        # Stores the predicted instance segmentation mask with threshold 0.5 ('IS' task)
                        detection['mask'] = masks[i][0, int(y1):int(y2), int(x1):int(x2)] >= 0.5

                    # Stores the predictions as a FiftyOne detection instance inside the detections list
                    detections.append(detection)

                # Adds the extracted model predictions to the original FiftyOne sample instance
                sample[f'prediction_{model_name}'] = fo.Detections(detections=detections)
                sample.save()

        # Runs the evaluation for the model predictions
        self.evaluate_model_predictions(f'prediction_{model_name}', detection_task, checkpoint['conf_thr'])

    def evaluate_model_predictions(self, field_name, detection_task, conf_thr):
        """
        Evaluates the model predictions, which are stored in a specific field within each FiftyOne sample instance.

        :param field_name: The name of the FiftyOne sample field, where the model predictions are stored.
        :param detection_task: String that specifies the detection task (OD or IS).
        :param conf_thr: The model specific confidence threshold for F1-score maximization.
        """
        # Filters out the low confidence detections
        high_conf_view = self.dataset.filter_labels(field_name, F('confidence') >= float(conf_thr), only_matches=False)

        # Runs the evaluation
        high_conf_view.evaluate_detections(
            field_name,
            gt_field='ground_truth',
            eval_key=f'eval_{field_name.replace("-", "_")}',
            iou=0.25,
            iscrowd='iscrowd',
            use_masks=False,
            tolerance=2
        )

    def upload_model_predictions_heatmaps(self, checkpoint_path, device_id, cfg):
        """
        Iterates over all samples from the FiftyOne CityPersons dataset and infers them into the given detection model,
        which is loaded from the checkpoint path. The model predictions are then stored within the same FiftyOne sample
        instances.

        :param checkpoint_path: String path to a checkpoint file, which contains the model weights.
        :param device_id: The integer ID of the GPU, which should be used for inference.
        """
        #model_name = checkpoint_path.split('/')[-2].split('_')[-1]  # Extracts the model name from the checkpoint path
        model_name = cfg['model_name']   # Extracts the model name from the checkpoint path

        detection_task = model_name.split('-')[-1]

        # TODO Loads the detection model and the weights from the checkpoint file
        checkpoint = torch.load(checkpoint_path)
        model, device = load_model({'model_name': model_name, 'gpu_id': device_id})
        model.load_state_dict(checkpoint['model'])
        model.eval()
        # register hooks on the feature maps
        feature_maps = []
        model.backbone.register_forward_hook(lambda module, input, output: feature_maps.append(output))
        heatmap_dir = f'{cfg["heatmap_path"]}/{cfg["dataset"]}/{cfg["model_name"]}'
        if not os.path.exists(heatmap_dir):
            os.makedirs(heatmap_dir)

        # Iterates over all dataset samples and infers them into the detection model to generate the predictions
        with fo.ProgressBar() as pb:
            for sample in pb(self.dataset):
                # Loads the sample image
                img = Image.open(sample.filepath)
                img, _ = self.transforms(img, None)
                c, h, w = img.shape

                # TODO Extracts the predictions from the model for the given sample
                preds = model([img.to(device)])[0]
                labels = preds['labels'].cpu().detach().numpy()
                scores = preds['scores'].cpu().detach().numpy()
                boxes = preds['boxes'].cpu().detach().numpy()
                if 'IS' in detection_task:
                    masks = preds['masks'].cpu().detach().numpy()

                detections = []  # Used for storing all FiftyOne detection instances for a single sample
                for i, label, score, box in zip(range(len(labels)), labels, scores, boxes):
                    # Extracts the predicted bbox coordinates
                    x1, y1, x2, y2 = box
                    rel_box = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]


                    # TODO Stores the predictions for the 'OD' detection task inside a FiftyOne detection instance
                    detection = fo.Detection(
                        label='human',
                        bounding_box=rel_box,
                        confidence=score
                    )


                    #if 'IS' in detection_task:
                        # Stores the predicted instance segmentation mask with threshold 0.5 ('IS' task)
                    #    detection['mask'] = masks[i][0, int(y1):int(y2), int(x1):int(x2)] >= 0.5

                    # Stores the predictions as a FiftyOne detection instance inside the detections list
                    detections.append(detection)

                # Adds the extracted model predictions to the original FiftyOne sample instance
                sample[f'prediction_{model_name}_heatmaps'] = fo.Detections(detections=detections)
                sample.save()
                # ------------------- creat binary mask prediction -------------------
                activations = feature_maps.pop()
                if isinstance(activations, torch.Tensor):
                    activations = OrderedDict([("0", activations)])
                activations = {k: v.detach().cpu() for k, v in activations.items()}

                heatmap_avg = torch.mean(activations['0'], 1)  # (1, H, W)

                resized_heatmap = torchvision.transforms.Resize(size=(h, w), antialias=True)(heatmap_avg)[0]
                heatmap_mask = torch.nn.Sigmoid()(resized_heatmap)
                heatmap_mask = heatmap_mask.cpu().numpy()*100  # (H, W)
                cv2.imwrite(f'{heatmap_dir}/{sample.file_name}.png', heatmap_mask)

        # TODO Runs the evaluation for the model predictions
        self.evaluate_model_predictions(f'prediction_{model_name}_heatmaps', detection_task, checkpoint['conf_thr'])

    def evaluate_model_predictions_heatmaps(self, field_name, detection_task, conf_thr):
        """
        Evaluates the model predictions, which are stored in a specific field within each FiftyOne sample instance.

        :param field_name: The name of the FiftyOne sample field, where the model predictions are stored.
        :param detection_task: String that specifies the detection task (OD or IS).
        :param conf_thr: The model specific confidence threshold for F1-score maximization.
        """
        # Filters out the low confidence detections
        high_conf_view = self.dataset.filter_labels(field_name, F('confidence') >= float(conf_thr), only_matches=False)

        # TODO Runs the evaluation
        high_conf_view.evaluate_detections(
            field_name,
            gt_field='ground_truth',
            eval_key=f'eval_{field_name.replace("-", "_")}',
            iou=0.25,
            iscrowd='iscrowd',
            use_masks=True,
            tolerance=2
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/eval_config.yaml',
                        help='Path to the config yaml file, which contains the evaluation parameters.')
    args = parser.parse_args()
    # Reads the yaml file, which contains the evaluation parameters
    with open(args.config) as yaml_file:
        cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

    print(fo.config)  # Logs the FiftyOne configurations to the console
    print(cfg)

    if cfg['dataset'] == 'KIA':
        # Loads a FiftyOne dataset instance of the KIA test dataset
        fo_kia = KIA2FiftyOne(cfg['data_path'])
        # Infers the samples from the dataset into the trained detection model to create predictions and evaluates them
        fo_kia.upload_model_predictions_heatmaps(cfg['checkpoint_path'], cfg['gpu_id'], cfg)
    elif cfg['dataset'] == 'CityPersons':
        # Loads a FiftyOne dataset instance of the CityPersons validation dataset (which in our case is the test data)
        fo_citypersons = CityPersons2FiftyOne(cfg['data_path'])
        # Infers the samples from the dataset into the trained detection model to create predictions and evaluates them
        fo_citypersons.upload_model_predictions_heatmaps(cfg['checkpoint_path'], cfg['gpu_id'], cfg)


if __name__ == '__main__':
    main()
