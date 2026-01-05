import statistics
import torch
import torchvision
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import pickle
import tqdm


def size_info_extractions(df):
    """
    Obtaining the size information of the extracted activations and print them out.
    :param df: a dataframe containing the extracted activations for each input image.
    The extracted activations are stored in a dict: The key is the instance id, and
    the value is the corresponding extracted activation numpy array with a shape of (C, H_e, W_e).
    :return: The averaged width and height of all the extracted activations.
    """
    width = []
    height = []
    instance_num = 0
    for extractions in df['extracted_feature_maps']:
        for idx in extractions:
            _, h, w = extractions[idx].shape
            instance_num += 1
            width.append(w)
            height.append(h)

    avg_width_extraction = round(statistics.mean(width))
    avg_height_extraction = round(statistics.mean(height))
    print(f'The total number of extractions(instances) is {instance_num}\n')
    print(f'The maximum width of the extractions is {max(width)}')
    print(f'The minimum width of the extractions is {min(width)}')
    print(f'The averaged width of the extractions is {avg_width_extraction}\n')
    print(f'The maximum height of the extractions is {max(height)}')
    print(f'The minimum height of the extractions is {min(height)}')
    print(f'The averaged height of the extractions is {avg_height_extraction}\n')
    print(f'The averaged aspect-ratio is {round(avg_width_extraction / avg_height_extraction, 2)}')

    return avg_width_extraction, avg_height_extraction


def resize_extractions(df, avg_w, avg_h):
    """
    Resize each of the extracted activations to the averaged size (C, avg_avg_h, avg_w).
    :param df: a dataframe containing the extracted activations for each input image.
    The extracted activations are stored in a dict: The key is the instance id, and
    the value is the corresponding extracted activation numpy array with a shape of (C, H_e, W_e)
    :param avg_w: The averaged width of all the extractions
    :param avg_h: The averaged height of all the extractions
    :return: A dataframe containing the resized extractions for all the input images
    """
    df_resized = pd.DataFrame(columns=['resized_extractions'])
    for idx_df, extractions in tqdm.tqdm(enumerate(df['extracted_feature_maps'])):
        resized_extractions = {}
        for idx in extractions:
            # convert the numpy array to tensor
            tensor_extra = torch.from_numpy(extractions[idx])
            # resize the extracted activation patches to the averaged shape
            resized_tensor = torchvision.transforms.Resize(size=(avg_h, avg_w))(tensor_extra).cpu().numpy()
            # store the resize extractions in the dict with the corresponding image index
            resized_extractions[idx] = resized_tensor
        # creat a dataframe for each of input image
        extra = pd.DataFrame(
            [[resized_extractions]],
            columns=['resized_extractions'],
            index=[df.index.values.tolist()[idx_df]]
        )
        # append the DF to the final dataframe
        df_resized = df_resized.append(extra)
    return df_resized


def split_resized_extractions(resized_df):
    """
    Process all the resized extractions including splitting, vertically stacking and flattening. Finally,
    a feature matrix is returned
    :param resized_df: A dataframe containing the resized extractions for all the input images
    :return: A feature matrix with shape (Number of samples, Number of features)
    """
    # a list to store all the resized extractions
    all_extraction = []
    for extraction in resized_df['resized_extractions']:
        for idx in extraction:
            # append every resized extraction to the list
            all_extraction.append(extraction[idx])
    # split the extraction to three parts: up, middle and down. Each of the split has a shape of (N, C, 1, 2)
    split1 = np.copy(np.array(all_extraction)[:, :, 0:1, :])  # upper part
    split2 = np.copy(np.array(all_extraction)[:, :, 1:2, :])  # middle part
    split3 = np.copy(np.array(all_extraction)[:, :, 2:3, :])  # down part
    # vertically stack all the splits
    merged_1 = np.vstack((split2, split3))
    all_split_extraction = np.vstack((split1, merged_1))  # shape (3*N, C, 1, 2)

    # reshape the extraction to a vector for following
    all_split_extraction_vector = all_split_extraction.reshape(all_split_extraction.shape[0], -1) # shape (3*N, C*2)

    return all_split_extraction_vector


def isolation_forest(X):
    """
    Training an Isolation Forest model with the given feature matrix. Saving the trained model in local.
    :param X: Input feature matrix with the shape (N_samples, N_features)
    """
    # split the whole dataset to training and test sets
    X_train, X_test = train_test_split(X, test_size=0.4, random_state=42)

    # fit the model
    clf = IsolationForest(random_state=0, n_jobs=-1)
    print(f'Training the Isolation Forest model...')
    clf.fit(X_train)
    # # save the model to disk
    filename = 'isolation_forest_model.sav'
    pickle.dump(clf, open(filename, 'wb'))
    print(f'The trained model have been saved!')
    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    # predict on training set
    y_pred_train = loaded_model.predict(X_train)
    # predict on test set
    y_pred_test = loaded_model.predict(X_test)
    # get the error rate of training and test sets
    n_error_train = y_pred_train[y_pred_train == -1].size
    print(f'error rate train: {n_error_train}/{len(X_train)} = {round(n_error_train / len(X_train) * 100, 3)}%')
    n_error_test = y_pred_test[y_pred_test == -1].size
    print(f'error rate test: {n_error_test}/{len(X_test)} = {round(n_error_test / len(X_test) * 100, 3)}%')

def main():
    # load the dataframe containing the extracted activation patches from disk
    df = pd.read_pickle('extracted_activation.pkl')
    # print size info of all the extractions
    print('the sizes of all the extractions are calculating...')
    avg_w, avg_h = size_info_extractions(df)
    # resize all the extractions to the averaged shape
    resized_extractions_df = resize_extractions(df, avg_w, avg_h)
    # get the feature matrix
    vectors = split_resized_extractions(resized_extractions_df)
    print(f'the shape of the feature matrix is {vectors.shape}')
    # train the outlier detector
    isolation_forest(vectors)


if __name__ == '__main__':
    main()