import os
import shutil
from os.path import join

import numpy as np
import pandas as pd
from get_distance import get_distance
from get_knn_count import get_knn_count

from collections import Counter
def my_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)




def get_risk_info(
    data_dir, data_sets, cnns, layers, elem_name_str, elems, csv_dir_str, k_list, note, client
):
    # Get risk elem
    for layer in layers:

        for cnn in cnns:
            print("=== geting risk_info of {}_{} ===".format(cnn, layer))

            elem_name = elem_name_str.format(cnn)
            csv_dir = csv_dir_str.format(data_dir, cnn)
            #print (csv_dir)
            targets_df = pd.read_csv(join(csv_dir, "targets_{}.csv".format(data_sets[0])))
            
            # Get the number of classes for each label
            num_classes = [targets_df[label].nunique() for label in targets_df.columns]

            # Display the number of classes for each label
            print("Number of classes for each label:", num_classes)
            
            num_class =sum(num_classes)
            #print(num_class)
            #num_class = (
            #    int(
            #        pd.read_csv(join(csv_dir, "targets_{}.csv".format(data_sets[0])), header=None)
            #        .to_numpy()
            #        .flatten()[-1]
            #    )
            #    + 1
            #)
            get_distance(layer, elem_name, num_class, csv_dir, elems, data_sets)
            #get_knn_count(
            #    k_list, layer, elem_name, num_class, csv_dir, "cosine", data_sets
            #)
           
    # Merge csv
    csv_path_list = []
    for cnn in cnns:
        for layer in layers:
            for elem in elems:
                if elem=='fangcha':
                    if layer=='x4':
                        continue
                    if cnn=='CCT':
                        continue
                if elem =='xs8' or elem=='xs1'or elem =='xs3' or elem=='xs5':
                        if layer == 'x4':
                            continue
                        if cnn == 'CCT':
                            continue
                if elem == 'paddingdis':
                    if cnn == 'CCT':
                        continue
                if elem == 'padknn8' or elem == 'padknn1':
                    if cnn == 'CCT':
                        continue
                if elem == 'xsdis':
                    if cnn == 'CCT':
                        continue
                    if layer == 'x4':
                        continue
                if elem == 'all3' or elem == 'all5':
                    if cnn == 'CCT':
                        continue
                    if layer == 'x4':
                        continue
                csv_path_list.append(
                        join(
                            csv_dir_str.format(data_dir, cnn),
                            "{}_{}_{}.csv".format(cnn, layer, elem),
                        )
                    )





    
    # Load the initial data with image names
    all_info = pd.read_csv(csv_path_list[0], header=None).to_numpy()[:, :2]
    
    # Append all other data columns from the CSV files
    for csv_path in csv_path_list:
        csv = pd.read_csv(csv_path, header=None).to_numpy()[:, 2:]
        all_info = np.hstack((all_info, csv))
    
    # Create a copy of the data for anonymization
    df_all_info = np.copy(all_info)
    
    # Create a dictionary mapping original image paths to their extracted IDs
    original_to_id = {}
    for line in all_info[1:]:  # Skip header if present
        original_path = line[0]
        image_id = original_path.split("/")[-1].split(".")[0]  # Extract the image ID
        original_to_id[original_path] = image_id
    
    # Create anonymized image names
    dataset_name = "BCNB"  # Change this to your preferred prefix
    anonymized_names = {}
    id_to_anonymized = {}
    counter = 1
    
    for original_path in original_to_id.keys():
        anonymized_name = f"{dataset_name}_{counter}"
        anonymized_names[original_path] = anonymized_name
        id_to_anonymized[original_to_id[original_path]] = anonymized_name
        counter += 1
    
    # Create mapping file (anonymized name -> original path)
    image_map = np.column_stack(([anonymized_names[path] for path in original_to_id.keys()], 
                                 list(original_to_id.keys())))
    
    # Save the mapping file
    map_path = f"/home/15t/Gul/SG/sheeraz/result_archive/{data_dir}{cnn}/{client}/all_data_map.csv"
    pd.DataFrame(image_map).to_csv(map_path, header=None, index=None)
    
    # Replace original names with anonymized names in the main data
    for i in range(1, len(df_all_info)):  # Skip header if present
        df_all_info[i, 0] = anonymized_names[all_info[i, 0]]
    
    # Save the anonymized all_info
    anonymized_path = f"/home/15t/Gul/SG/sheeraz/result_archive/risk_elem/{data_dir}/{client}/risk_dataset{note}/all_data_info.csv"
    pd.DataFrame(df_all_info).to_csv(anonymized_path, header=None, index=None)
    
    
    # Save the original all_info
    #pd.DataFrame(all_info).to_csv(
    #    f"/home/Gul/SG/sheeraz/result_archive/risk_elem/{data_dir}/{client}/risk_dataset{note}/all_data_info.csv",
    #    header=None,
    #    index=None,
    #)
    
    # Load the image IDs for each set
    train_ids = np.loadtxt('/home/15t/Gul/Datasets/BCNB/dataset-splitting/train_id.txt', dtype=str)
    val_ids = np.loadtxt('/home/15t/Gul/Datasets/BCNB/dataset-splitting/val_id.txt', dtype=str)
    test_ids = np.loadtxt('/home/15t/Gul/Datasets/BCNB/dataset-splitting/test_id.txt', dtype=str)
    
    # Create a dictionary to map image IDs to their corresponding set
    id_to_set = {}
    for image_id in train_ids:
        id_to_set[image_id] = 'train'
    for image_id in val_ids:
        id_to_set[image_id] = 'val'
    for image_id in test_ids:
        id_to_set[image_id] = 'test'
    
    # Print initial ID counts for verification
    print(f"Train IDs: {len(train_ids)}")
    print(f"Val IDs: {len(val_ids)}")
    print(f"Test IDs: {len(test_ids)}")
    
    # Create anonymized version of the headers
    anonymized_header = df_all_info[0].copy()
    
    # Process each data set using anonymized names
    for data_set in ['train', 'val', 'test']:
        temp_csv = [anonymized_header]  # Use anonymized header
        
        for i in range(1, len(all_info)):
            original_path = all_info[i, 0]
            image_id = original_to_id[original_path]
            
            if id_to_set.get(image_id, None) == data_set:
                # Add the corresponding anonymized row to the temp_csv
                temp_csv.append(df_all_info[i])
        
        print(f"Saved {data_set} with {len(temp_csv) - 1} records")
        
        pd.DataFrame(temp_csv).to_csv(
            f"/home/15t/Gul/SG/sheeraz/result_archive/risk_elem/{data_dir}/{client}/risk_dataset{note}/{data_set}.csv",
            header=None,
            index=None,
        )
     
    '''    
    # Load the initial data with image names
    all_info = pd.read_csv(csv_path_list[0], header=None).to_numpy()[:, :2]
    
    # Append all other data columns from the CSV files
    for csv_path in csv_path_list:
        csv = pd.read_csv(csv_path, header=None).to_numpy()[:, 2:]
        all_info = np.hstack((all_info, csv))
    
    # Create a copy of the data for anonymization
    df_all_info = np.copy(all_info)
    
    # Create a dictionary mapping original image paths to their filenames
    original_to_filename = {}
    for line in all_info[1:]:  # Skip header if present
        original_path = line[0]
        filename = original_path.split("/")[-1]  # Extract just the filename
        original_to_filename[original_path] = filename
    
    # Create anonymized image names
    dataset_name = "BRCA"  # Change this to your preferred prefix
    anonymized_names = {}
    filename_to_anonymized = {}
    counter = 1
    
    for original_path in original_to_filename.keys():
        anonymized_name = f"{dataset_name}_{counter}"
        anonymized_names[original_path] = anonymized_name
        filename_to_anonymized[original_to_filename[original_path]] = anonymized_name
        counter += 1
    
    # Create mapping file (anonymized name -> original path)
    image_map = np.column_stack(([anonymized_names[path] for path in original_to_filename.keys()], 
                                 list(original_to_filename.keys())))
    
    # Save the mapping file
    map_path = f"/home/15t/Gul/SG/sheeraz/result_archive/{data_dir}{cnn}/{client}/all_data_map.csv"
    pd.DataFrame(image_map).to_csv(map_path, header=None, index=None)
    
    # Replace original names with anonymized names in the main data
    for i in range(1, len(df_all_info)):  # Skip header if present
        df_all_info[i, 0] = anonymized_names[all_info[i, 0]]
    
    # Save the anonymized all_info
    anonymized_path = f"/home/15t/Gul/SG/sheeraz/result_archive/risk_elem/{data_dir}/{client}/risk_dataset{note}/all_data_info.csv"
    pd.DataFrame(df_all_info).to_csv(anonymized_path, header=None, index=None)
    
    # Save the original all_info
    #pd.DataFrame(all_info).to_csv(
    #    f"/home/15t/Gul/SG/sheeraz/result_archive/risk_elem/{data_dir}/{client}/risk_dataset{note}/all_data_info.csv",
    #    header=None,
    #    index=None,/home/15t/Gul/SG/sheeraz/result_archive/Multilabel_Low50/
    #)
    
    # Load the IDs and append ".jpg" to each
    train_ids = np.loadtxt('/home/15t/Gul/Datasets/Combined_BRCA/dataset-split/train_ids.txt', dtype=str)
    val_ids = np.loadtxt('/home/15t/Gul/Datasets/Combined_BRCA/dataset-split/val_ids.txt', dtype=str)
    test_ids = np.loadtxt('/home/15t/Gul/Datasets/Combined_BRCA/dataset-split/test_ids.txt', dtype=str)
    
    # Append ".jpg" to match actual file names
    train_ids = [f"{img_id}.jpg" for img_id in train_ids]
    val_ids = [f"{img_id}.jpg" for img_id in val_ids]
    test_ids = [f"{img_id}.jpg" for img_id in test_ids]
    
    # Print initial ID counts for verification
    print(f"Train IDs: {len(train_ids)}")
    print(f"Val IDs: {len(val_ids)}")
    print(f"Test IDs: {len(test_ids)}")
    
    # Output directory
    output_base_path = f"/home/15t/Gul/SG/sheeraz/result_archive/risk_elem/{data_dir}/{client}/risk_dataset{note}"
    
    # Create anonymized version of the header
    anonymized_header = df_all_info[0].copy()
    
    # Process train, val, and test separately
    for data_set, dataset_ids in zip(["train", "val", "test"], [train_ids, val_ids, test_ids]):
        temp_csv = [anonymized_header]  # Start with the anonymized header
        
        for img_id in dataset_ids:
            # Collect all matching records for this ID
            matching_records = []
            matching_indices = []
            
            for i in range(1, len(all_info)):
                file_name = all_info[i, 0].split("/")[-1]
                if file_name == img_id:
                    matching_records.append(all_info[i])
                    matching_indices.append(i)
            
            # If we found any matches, add one to our output (using the anonymized version)
            if matching_records:
                # Get the index of the record we want to use (the last one in this case)
                idx_to_use = matching_indices[-1]
                # Add the corresponding anonymized row to the output
                temp_csv.append(df_all_info[idx_to_use])
        
        # Save to CSV
        output_path = f"{output_base_path}/{data_set}.csv"
        pd.DataFrame(temp_csv).to_csv(output_path, header=None, index=None)
        
        print(f"? Saved {output_path} with {len(temp_csv) - 1} records")           
    
    '''
    '''
    all_info = pd.read_csv(csv_path_list[0], header=None).to_numpy()[:, :2]

    for csv_path in csv_path_list:
        csv = pd.read_csv(csv_path, header=None).to_numpy()[:, 2:]
       #print(csv_path)
        all_info = np.hstack((all_info, csv))
    
    
    pd.DataFrame(all_info).to_csv(
        "/home/15t/Gul/SG/sheeraz/result_archive/risk_elem/{}/{}/risk_dataset{}/all_data_info.csv".format( #change path to the save distribution
            data_dir, client, note
        ),
        header=None,
        index=None,
    )

    all_info = all_info.tolist()
    print("the length of all info is: ")
    print(len(all_info))
    
    # Load the image IDs for each set
    # Load the image IDs for each set
    train_ids = np.loadtxt('/home/Gul/Datasets/BCNB/BCNB/dataset-splitting/train_id.txt', dtype=str)
    val_ids = np.loadtxt('/home/Gul/Datasets/BCNB/BCNB/dataset-splitting/val_id.txt', dtype=str)
    test_ids = np.loadtxt('/home/Gul/Datasets/BCNB/BCNB/dataset-splitting/test_id.txt', dtype=str)
    
    # Create a dictionary to map image IDs to their corresponding set
    id_to_set = {}
    for image_id in train_ids:
        id_to_set[image_id] = 'train'
    for image_id in val_ids:
        id_to_set[image_id] = 'val'
    for image_id in test_ids:
        id_to_set[image_id] = 'test'
    
    # Print initial ID counts for verification
    print(f"Train IDs: {len(train_ids)}")
    print(f"Val IDs: {len(val_ids)}")
    print(f"Test IDs: {len(test_ids)}")
    for data_set in ['train', 'val', 'test']:
        temp_csv = [all_info[0]]
        for line in all_info[1:]:
            image_id = line[0].split("/")[-1].split(".")[0]  # Extract the image ID from the file path
            #print(image_id)
            if id_to_set.get(image_id, None) == data_set:
                temp_csv.append(line)
        print(f"? Saved {data_set} with {len(temp_csv) - 1} records")
        pd.DataFrame(temp_csv).to_csv(
            "/home/Gul/SG/sheeraz/result_archive/risk_elem/{}/{}/risk_dataset{}/{}.csv".format(
                data_dir, client, note, data_set
            ),
            header=None,
            index=None,
        )
    
    '''
    '''
    # Load the IDs and append ".jpg" to each
    train_ids = np.loadtxt('/home/Gul/Datasets/Combined_BRCA/dataset-split/train_ids.txt', dtype=str)
    val_ids = np.loadtxt('/home/Gul/Datasets/Combined_BRCA/dataset-split/val_ids.txt', dtype=str)
    test_ids = np.loadtxt('/home/Gul/Datasets/Combined_BRCA/dataset-split/test_ids.txt', dtype=str)
    
    # Append ".jpg" to match actual file names
    train_ids = [f"{img_id}.jpg" for img_id in train_ids]
    val_ids = [f"{img_id}.jpg" for img_id in val_ids]
    test_ids = [f"{img_id}.jpg" for img_id in test_ids]
    
    # Print initial ID counts for verification
    print(f"Train IDs: {len(train_ids)}")
    print(f"Val IDs: {len(val_ids)}")
    print(f"Test IDs: {len(test_ids)}")
    
    # Output directory
    output_base_path = f"/home/Gul/SG/sheeraz/result_archive/risk_elem/{data_dir}/{client}/risk_dataset{note}"
    
    # Process train, val, and test separately
    for data_set, dataset_ids in zip(["train", "val", "test"], [train_ids, val_ids, test_ids]):
        temp_csv = [all_info[0]]  # Start with the header
        
        for img_id in dataset_ids:
            # Collect all matching records for this ID
            matching_records = []
            for line in all_info[1:]:
                file_name = line[0].split("/")[-1]
                if file_name == img_id:
                    matching_records.append(line)
            
            # If we found any matches, add one to our output
            if matching_records:
                # Here you can implement your logic to choose which record to use
                # For example, you might want to check some values in the records
                # For now, let's say we want the last record (most recent/updated)
                temp_csv.append(matching_records[-1])  # Using the last matching record instead of first
        
        # Save to CSV
        output_path = f"{output_base_path}/{data_set}.csv"
        pd.DataFrame(temp_csv).to_csv(output_path, header=None, index=None)
        
        print(f"? Saved {output_path} with {len(temp_csv) - 1} records")
                    
    '''
    #print(all_info)
    #for data_set in data_sets:
    #    #print(data_set)
    #    temp_csv = [all_info[0]]
    #    #print(temp_csv)
    #    for line in all_info[1:]:
    #        #print(line[0])
    #       #print(line[0].split("/")[1])
    #        if line[0].split("/")[1] == data_set:
    #            temp_csv.append(line)
    #    print(temp_csv)
    #    pd.DataFrame(temp_csv).to_csv(
    #        "/home/ssd0/SG/sheeraz/result_archive/risk_elem/{}/risk_dataset{}/{}.csv".format( #change path to the save distribution
    #            data_dir, note, data_set
    #        ),
    #        header=None,
    #        index=None,
    #    )


# get risk elem
if __name__ == "__main__":

    get_risk_info()
