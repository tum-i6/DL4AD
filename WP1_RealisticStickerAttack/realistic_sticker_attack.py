from datetime import datetime
from random import randrange
import argparse
import shutil
import os
import sys

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import trange, tqdm
import numpy as np
import torch.nn as nn
import torch
import cv2
import yaml

import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/realistic_sticker_attack_config.yaml',
                        help='Path to the config yaml file, which contains the Realistic Sticker Attack parameters.')
    args = parser.parse_args()

    # Reads the yaml file, which contains the Realistic Sticker Attack parameters
    with open(args.config) as yaml_file:
        cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Loads the classification model and the device
    model, device = utils.load_model(cfg['model_name'], cfg['device_ids'], cfg['weights_path'])
    model.eval()
    # Assigns the full name for the selected Realistic Sticker Attack algorithm
    attack_algorithm = utils.get_attack_algorithm_name(cfg['attack_algorithm'])
    for class_id in cfg['true_target_classes']:
        model_variant_name=cfg['weights_path'].split('/')[-2]
        tqdm.write(f"Experiments on class {class_id}")
        tqdm.write(f"Initiating {attack_algorithm} ({cfg['attack_algorithm']}) for attacking {cfg['model_name']} on class "
                   f"{class_id}.")
        if cfg['attack_target_class'] in range(43):
            tqdm.write(f"Attack goal set to target attack optimization towards class {cfg['attack_target_class']}.")
        else:
            tqdm.write(f"Attack goal set to misclassification of class {class_id}.")

        # Defines the experiment name and creates a new sub-folder for storing the training results and TensorBoard logs
        experiment_name = f"{str(datetime.now())[:-7].replace(' ', '_').replace(':', '-')}_{cfg['model_name']}_{cfg['attack_algorithm']}_on_class_{class_id}"
        experiment_name = f"{model_variant_name}/{cfg['attack_algorithm']}_on_class_{class_id}"
        if cfg['attack_target_class'] in range(43):
            experiment_name += f"_attack_target_class_{cfg['attack_target_class']}"
        experiment_path = f"{cfg['output_path']}/{experiment_name}"
        if not os.path.exists(f"{experiment_path}/sticker_overlays"):
            os.makedirs(f"{experiment_path}/sticker_overlays")

        # Displays all parameters for the current experiment and creates a copy of the respective yaml file
        shutil.copyfile(args.config, experiment_path + '/config.yaml')
        tqdm.write('\nPARAMETERS:')
        for param in cfg.keys():
            tqdm.write('    ' + param + ' = ' + str(cfg[param]))



        criterion = nn.CrossEntropyLoss()  # Defines the loss function for calculating the prediction loss

        toTensor = transforms.ToTensor()  # Used for converting images to torch tensors
        # Used for randomly enhancing the sticker brightness, contrast, saturation and hue values
        random_enhance = transforms.ColorJitter(
            brightness=(0.7, 1.3),
            contrast=(0.7, 1.3),
            saturation=(0.7, 1.3),
            hue=(-0.05, 0.05)
        )

        # Loads the train and validation images
        train_images = utils.load_images(f"{cfg['data_path']}/{class_id}/{'train_set'}", toTensor, device)
        val_images = utils.load_images(f"{cfg['data_path']}/{class_id}/{'validation_set'}", toTensor, device)

        # Loads the image properties for the train and validation images
        train_img_properties = utils.get_image_properties(train_images, device)
        val_img_properties = utils.get_image_properties(val_images, device)

        # Preprocesses the train and validation images for model inference
        clean_train_images = utils.preprocess_images(train_images, device)
        clean_val_images = utils.preprocess_images(val_images, device)

        # Loads the true train and validation targets used for calculating the prediction loss
        train_true_targets = ((torch.ones((len(train_images)))*class_id).type(torch.LongTensor)).to(device)
        val_true_targets = ((torch.ones((len(val_images)))*class_id).type(torch.LongTensor)).to(device)

        # Loads the train and validation targets used for calculating the prediction loss for the target attack
        train_attack_targets = None
        val_attack_targets = None
        if cfg['attack_target_class'] in range(43):
            train_attack_targets = \
                ((torch.ones((len(train_images))) * cfg['attack_target_class']).type(torch.LongTensor)).to(device)
            val_attack_targets = \
                ((torch.ones((len(val_images))) * cfg['attack_target_class']).type(torch.LongTensor)).to(device)

        # Infers the clean train and validation data into the model to log the prediction performance on clean images
        display_text = '\nBaseline Prediction Performance on Clean Images'
        clean_accuracies, clean_losses = \
            utils.get_prediction_performance(
                model,
                criterion,
                clean_train_images,
                (train_true_targets, train_attack_targets),
                clean_val_images,
                (val_true_targets, val_attack_targets),
                display_text
            )

        # Infers the clean train and validation data into the model to calculate the clean saliency maps and heatmaps
        clean_avg_train_saliency, clean_train_heatmaps, clean_train_individual_heatmaps = \
            utils.get_saliency_maps(model, clean_train_images, (train_true_targets, train_attack_targets))
        clean_avg_val_saliency, clean_val_heatmaps, clean_val_individual_heatmaps = \
            utils.get_saliency_maps(model, clean_val_images, (val_true_targets, val_attack_targets))

        # Loads the sticker texture pool
        stickers = utils.load_images(cfg['stickers_path'], toTensor, device, color_format='RGBA')

        # Loads the attack mask, used for constraining the sticker placement
        mask = utils.load_mask(f"{cfg['masks_path']}/{class_id}.png", toTensor, device)
        sign_area = torch.sum(mask[0])  # Calculates the mask area to use it as the representative sign surface area

        # Loads the average saliency map of the train set (used for positioning the stickers in case of the SSA algorithm)
        # The saliency map is filtered using the mask to avoid placements outside the mask
        ssa_saliency = torch.mul(clean_avg_train_saliency, mask)

        # Transparent overlay used as the final sticker attack overlay (used for sticker placement)
        final_sticker_overlay = torch.zeros((4, clean_train_images.shape[2], clean_train_images.shape[3])).to(device)

        # Instantiates TensorBoard SummaryWriters for logging the sticker attack performance to TensorBoard
        writer_train = SummaryWriter(experiment_path + '/TensorBoard_Logs/Train')
        writer_val = SummaryWriter(experiment_path + '/TensorBoard_Logs/Val')
        writer_model_baseline = SummaryWriter(experiment_path + '/TensorBoard_Logs/Baseline')

        sticker_performances = {}  # Used for storing the attack performance for each sticker overlay

        used_stickers = []  # Used for tracking the IDs of all textures that were applied to the final sticker overlay
        tqdm.write(f"Experiments on class {class_id}")
        tqdm.write(f"\n{attack_algorithm} initiated!")
        # Searches for the best single sticker overlay with the highest validation loss on the true target class in case of
        # the misclassification optimization, or the lowest validation loss on the attack target class in case of the
        # attack target optimization
        # In the next iteration, this single sticker overlay is used to find the best double sticker overlay with the same
        # approach
        # This process is repeated max_sticker_amount times
        for sticker_amount in range(1, cfg['max_sticker_amount'] + 1):
            if cfg['attack_target_class'] in range(43):
                # Used for tracking the sticker overlay with the lowest validation loss on the attack target class
                best_val_loss = np.Inf
            else:
                # Used for tracking the sticker overlay with the highest validation loss on the true target class
                best_val_loss = -np.Inf

            # Placeholders for storing the train and validation performance for the strongest sticker attack
            # Initialized with the baseline train and validation performance on clean images
            best_accuracies = clean_accuracies
            best_losses = clean_losses

            # The search for the best sicker is repeated num_epochs times
            tqdm.write(f"Experiments on class {class_id}")
            for epoch in range(cfg['num_epochs']):
                with torch.no_grad():
                    # Instantiates the visual sticker texture loading bar and logs the current prediction performance
                    loading_bar_desc = \
                        f"{model_variant_name}-{cfg['attack_algorithm']} [Class:{class_id},Stickers:{sticker_amount}/" \
                        f"{cfg['max_sticker_amount']},Epoch:{epoch+1}/{cfg['num_epochs']}] Stickers"
                    texture_iterator = trange(len(stickers), desc=loading_bar_desc, leave=True, file=sys.stdout)
                    utils.update_texture_iterator_postfix(texture_iterator, sticker_amount, best_accuracies, best_losses)
                    # Iterates over all sticker textures to find the best performing one
                    for i in texture_iterator:
                        # The Random Sticker Application algorithm samples a random sticker texture and does not iterate
                        # further
                        if attack_algorithm == 'Random Sticker Application':
                            while True:
                                i = randrange(0, len(stickers))
                                if cfg['repeating_stickers'] or (i not in used_stickers):
                                    break

                        # The current texture will be ignored if it has been already applied to the final sticker overlay
                        # and repeating stickers is disabled
                        if (not cfg['repeating_stickers']) and (i in used_stickers):
                            continue

                        sticker = stickers[i].clone()  # Loads the sticker texture
                        test_overlay = final_sticker_overlay.clone()  # Used for evaluating the current sticker texture

                        # Resizes the sticker randomly
                        sticker = \
                            utils.random_resize(
                                sticker,
                                cfg['min_sticker_length'],
                                cfg['max_sticker_length'],
                                cfg['min_sticker_area'],
                                cfg['max_sticker_area'],
                                cfg['min_sticker_ratio'],
                                cfg['max_sticker_ratio']
                            )

                        sticker = utils.random_rotate(sticker, cfg['max_rotation_angle'])  # Rotates the sticker randomly

                        # Samples random x & y coordinates for positioning the sticker on the sign surface
                        # The MCSA and RSA attack applies the sticker to a random position on the sign surface
                        if attack_algorithm == 'Monte Carlo Sticker Application' or attack_algorithm == 'Random Sticker Application':
                            x, y = utils.get_random_position(sticker, mask, test_overlay)
                        # The Saliency Sticker Application (SSA) applies the sticker to the strongest saliency points
                        elif attack_algorithm == 'Saliency Sticker Application':
                            x, y = utils.get_highest_saliency_position(sticker, ssa_saliency)

                        # Randomly enhances sticker brightness, contrast, saturation and hue values
                        sticker[0:3] = random_enhance(sticker[0:3])

                        # Applies the sticker to the test overlay
                        test_overlay = utils.sticker2overlay(test_overlay, sticker, x, y)
                        # Cuts stickers that are outside the mask
                        test_overlay[0:3] = torch.mul(test_overlay[0:3], mask)
                        test_overlay[3] = torch.mul(test_overlay[3], mask[0])

                        # Extracts the non transparent area of the sticker overlay
                        overlay_mask = test_overlay[3] != 0
                        test_overlay[0:3, ~overlay_mask] = 0  # Sets all pixel values of the transparent area to zero

                        # Calculates the area of the sticker overlay, to get the sign coverage in percent
                        sticker_overlay_area = torch.sum(overlay_mask)
                        sign_coverage = sticker_overlay_area / sign_area * 100

                        # Applies the sticker overlay to the train and validation data
                        attacked_train_images = \
                            utils.sticker2images(
                                clean_train_images.clone(),
                                train_img_properties,
                                test_overlay,
                                overlay_mask,
                                device
                            )
                        attacked_val_images = \
                            utils.sticker2images(
                                clean_val_images.clone(),
                                val_img_properties,
                                test_overlay,
                                overlay_mask,
                                device
                            )

                        # Infers the attacked train and validation data into the model to log the prediction performance
                        display_text = f"\nPrediction Performance on Attacked Images [Sign Coverage: {sign_coverage:.1f}%]"
                        accuracies, losses = \
                            utils.get_prediction_performance(
                                model,
                                criterion,
                                attacked_train_images,
                                (train_true_targets, train_attack_targets),
                                attacked_val_images,
                                (val_true_targets, val_attack_targets),
                                display_text
                            )

                        # Checks if the current sticker overlay achieved a better performance
                        new_best = False
                        if cfg['attack_target_class'] in range(43):
                            if losses['val_losses'][1] < best_val_loss:
                                new_best_val_loss = losses['val_losses'][1]
                                loss_change = 'decreased'
                                new_best = True
                        else:
                            if losses['val_losses'][0] > best_val_loss:
                                new_best_val_loss = losses['val_losses'][0]
                                loss_change = 'increased'
                                new_best = True

                        if new_best:
                            tqdm.write(
                                f"- Validation loss {loss_change}] ({best_val_loss:.6f} --> {new_best_val_loss:.6f})."
                                f"Saving new sticker overlay...")
                            utils.update_texture_iterator_postfix(texture_iterator, sticker_amount, accuracies, losses)
                            best_val_loss = new_best_val_loss

                            # Stores all sticker properties for the current sticker overlay
                            best_sticker_overlay = test_overlay.clone()
                            best_sticker_id = i
                            best_sign_coverage = sign_coverage
                            best_accuracies = accuracies
                            best_losses = losses
                            best_attacked_train_images = attacked_train_images.clone()
                            best_attacked_val_images = attacked_val_images.clone()

                        # The Random Sticker Application algorithm samples only one sticker texture and does not iterate
                        # further
                        if attack_algorithm == 'Random Sticker Application':
                            break
                    if attack_algorithm == 'Random Sticker Application':
                        break

            # Infers the attacked train and validation data into the model to calculate the new saliency maps and heatmaps
            attacked_avg_train_saliency, attacked_train_heatmaps, attacked_train_individual_heatmaps = \
                utils.get_saliency_maps(model, best_attacked_train_images, (train_true_targets, train_attack_targets))
            attacked_avg_val_saliency, attacked_val_heatmaps, attacked_val_individual_heatmaps = \
                utils.get_saliency_maps(model, best_attacked_val_images, (val_true_targets, val_attack_targets))

            # Erases the image regions of the saliency map, which are now covered by the sticker
            covered_spots = best_sticker_overlay[3] == 0
            if attack_algorithm == 'Saliency Sticker Application':
                # In case of SSA, cover bigger parts of the saliency map to avoid overlapping sticker placements
                kernel = np.ones((5, 5), np.uint8)
                covered_spots = cv2.erode(np.array(covered_spots.to('cpu')).astype('uint8'), kernel, iterations=7)
                covered_spots = toTensor(covered_spots).to(device)
            ssa_saliency *= covered_spots  # Covers the saliency map

            # Adds the sticker id to the used_stickers list
            used_stickers.append(best_sticker_id)

            # Applies the best sticker to the final sticker overlay
            final_sticker_overlay = best_sticker_overlay

            # Stores the best sticker overlay as a .png file
            utils.save_sticker_overlay(
                cfg['model_name'],
                cfg['attack_algorithm'],
                cfg['attack_target_class'],
                experiment_path,
                best_sticker_overlay,
                sticker_amount,
                best_accuracies,
                best_sign_coverage
            )

            # Stores the sticker performance for the current sticker overlay inside the performance dictionary
            utils.save_sticker_performance(
                sticker_performances,
                sticker_amount,
                best_sign_coverage,
                cfg['attack_target_class'],
                best_accuracies,
                best_losses
            )

            # Visualizes the Realistic Sticker Attack in TensorBoard
            utils.visualize_attack2TensorBoard(
                writer_train,
                sticker_amount,
                cfg['attack_target_class'],
                device,
                best_sticker_overlay,
                mask,
                clean_avg_train_saliency,
                ssa_saliency,
                attacked_train_heatmaps,
                attacked_val_heatmaps,
                attacked_train_individual_heatmaps,
                attacked_val_individual_heatmaps
            )

            # Logs the prediction performance for the current sticker overlay to TensorBoard
            utils.scalars_attack2TensorBoard(
                writer_train,
                writer_val,
                sticker_amount,
                cfg['attack_target_class'],
                best_sign_coverage,
                best_accuracies,
                best_losses
            )

            # Logs the baseline prediction performance on clean images to TensorBoard
            utils.scalars_baseline2TensorBoard(
                writer_model_baseline,
                sticker_amount,
                cfg['attack_target_class'],
                clean_accuracies,
                clean_losses
            )

        # Visualizes the baseline Realistic Sticker Attack in TensorBoard
        utils.visualize_baseline2TensorBoard(
            writer_train,
            cfg['attack_target_class'],
            device,
            mask,
            clean_avg_train_saliency,
            clean_train_heatmaps,
            clean_val_heatmaps,
            clean_train_individual_heatmaps,
            clean_val_individual_heatmaps
        )

        # Logs the performance dictionary and the experiment hyperparameters to TensorBoard
        #writer_train.add_hparams(cfg, sticker_performances)
        writer_train.close()

        print(f"\n{attack_algorithm} finished.")


if __name__ == '__main__':
    main()
