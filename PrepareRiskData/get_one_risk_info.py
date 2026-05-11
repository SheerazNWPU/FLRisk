from os.path import join

import numpy as np
import pandas as pd
from get_one_distance import get_one_distance
from get_one_knn_count import get_one_knn_count


def get_one_risk_info(
    data_dir, data_sets, cnns, layers, elem_name_str, elems, csv_dir_str, k_list, client
):
    # get risk elem
    for layer in layers:
        for cnn in cnns:
            print("=== geting one_risk_info of {}_{} ===".format(cnn, layer))

            elem_name = elem_name_str.format(cnn)
            csv_dir = csv_dir_str.format(data_dir, cnn)
            # Read the targets CSV file
            targets_df = pd.read_csv(join(csv_dir, "targets_{}.csv".format(data_sets[0])))
            
            # Get the number of classes for each label
            num_classes = [targets_df[label].nunique() for label in targets_df.columns]

            # Display the number of classes for each label
            print("Number of classes for each label:", num_classes)
            
            num_class =sum(num_classes)
            #df1 = pd.read_csv(join(csv_dir, "targets_{}.csv".format(data_sets[0])), header=None)
            #num_class = len(df1.columns)
            #print(num_class)
             
            
            #num_class = (
            #    int(
            #        pd.read_csv(join(csv_dir, "targets_{}.csv".format(data_sets[0])), header=None)
            #        .to_numpy()
            #        .flatten()[-1]
            #    )
            #    + 1
            #)
            #print(pd.read_csv(join(csv_dir, "targets_{}.csv".format(data_sets[0])), header=None)
            #        .to_numpy())
            
            get_one_distance(layer, elem_name, num_class, targets_df, csv_dir, elems, data_sets)
            #get_one_knn_count(
            #    k_list, layer, elem_name, num_class, targets_df,  csv_dir, "cosine", data_sets
            #)

    # Merge csv
    csv_path_list = []
    for cnn in cnns:
        for layer in layers:
            for elem in elems:
                
                if elem == 'fangcha':
                    if layer == 'x4':
                        continue
                    if cnn == 'CCT':
                        continue
                if elem == 'xs8' or elem == 'xs1' or elem == 'xs3' or elem == 'xs5':
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
                if elem=='xsdis':
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
                        "{}_{}_one_{}.csv".format(cnn, layer, elem),
                    )
                )
    
    # Load the initial data with image names
    all_info = pd.read_csv(csv_path_list[0], header=None).to_numpy()[:, :2]
    
    # Append all other data columns from the CSV files
    for csv_path in csv_path_list:
        print(csv_path)
        csv = pd.read_csv(csv_path, header=None).to_numpy()[:, -1:]
        all_info = np.hstack((all_info, csv))
    
    # Create a copy of the data for anonymization
    df_all_info = np.copy(all_info)
    
    # Create a dictionary mapping original image paths to their extracted IDs
    original_to_id = {}
    for i in range(1, len(all_info)):  # Skip header if present
        original_path = all_info[i, 0]
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
    map_path = f"/home/15t/Gul/SG/sheeraz/result_archive/risk_elem/{data_dir}/{client}/DBLP-Scholar/pair_info_map.csv"
    pd.DataFrame(image_map).to_csv(map_path, header=None, index=None)
    
    # Replace original names with anonymized names in the main data
    for i in range(1, len(df_all_info)):  # Skip header if present
        if all_info[i, 0] in anonymized_names:
            df_all_info[i, 0] = anonymized_names[all_info[i, 0]]
    
    # Save the anonymized version
    pd.DataFrame(df_all_info).to_csv(
        f"/home/15t/Gul/SG/sheeraz/result_archive/risk_elem/{data_dir}/{client}/DBLP-Scholar/pair_info_more.csv",
        header=None,
        index=None,
    )
    # Save the original data (your original code)
    #pd.DataFrame(all_info).to_csv(
    #    f"/home/Gul/SG/sheeraz/result_archive/risk_elem/{data_dir}/{client}/DBLP-Scholar/pair_info_more.csv",
    #    header=None,
    #    index=None,
    #)
    
    # Convert to list for further processing
    all_info_list = all_info.tolist()
    df_all_info_list = df_all_info.tolist()
    
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
    
    # Process only the 'train' set as in your original code
    for data_set in ['train']:
        # Use the anonymized header
        temp_csv = [df_all_info_list[0]]
        
        # For each line in the original data
        for i in range(1, len(all_info_list)):
            line = all_info_list[i]
            image_id = line[0].split("/")[-1].split(".")[0]  # Extract the image ID from the file path
            
            if id_to_set.get(image_id, None) == data_set:
                # Add the corresponding anonymized row
                temp_csv.append(df_all_info_list[i])
        
        # Save the anonymized train dataset
        pd.DataFrame(temp_csv).to_csv(
            f"/home/15t/Gul/SG/sheeraz/result_archive/risk_elem/{data_dir}/{client}/DBLP-Scholar/325/{data_set}.csv",
            header=None,
            index=None,
        )

    
    '''
    # Load the initial data with image names
    all_info = pd.read_csv(csv_path_list[0], header=None).to_numpy()[:, :2]
    
    # Append all other data columns from the CSV files
    for csv_path in csv_path_list:
        print(csv_path)
        csv = pd.read_csv(csv_path, header=None).to_numpy()[:, -1:]
        all_info = np.hstack((all_info, csv))
    
    # Create a copy of the data for anonymization
    df_all_info = np.copy(all_info)
    
    # Create a dictionary mapping original image paths to their extracted IDs
    original_to_id = {}
    for i in range(1, len(all_info)):  # Skip header if present
        original_path = all_info[i, 0]
        image_id = original_path.split("/")[-1].split(".")[0]  # Extract the image ID
        original_to_id[original_path] = image_id
    
    # Create anonymized image names
    dataset_name = "BRCA"  # Change this to your preferred prefix
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
    map_path = f"/home/15t/Gul/SG/sheeraz/result_archive/risk_elem/{data_dir}/{client}/DBLP-Scholar/pair_info_map.csv"
    pd.DataFrame(image_map).to_csv(map_path, header=None, index=None)
    
    # Replace original names with anonymized names in the main data
    for i in range(1, len(df_all_info)):  # Skip header if present
        if all_info[i, 0] in anonymized_names:
            df_all_info[i, 0] = anonymized_names[all_info[i, 0]]
    
    # Save the original pair_info_more.csv
    #pd.DataFrame(all_info).to_csv(
    #    f"/home/15t/Gul/SG/sheeraz/result_archive/risk_elem/{data_dir}/{client}/DBLP-Scholar/pair_info_more.csv",
    #    header=None,
    #    index=None,
    #)
    
    # Save the anonymized pair_info_more.csv
    pd.DataFrame(df_all_info).to_csv(
        f"/home/15t/Gul/SG/sheeraz/result_archive/risk_elem/{data_dir}/{client}/DBLP-Scholar/pair_info_more.csv",
        header=None,
        index=None,
    )
    
    # Load the image IDs for each set (using Combined_BRCA dataset)
    train_ids = np.loadtxt('/home/15t/Gul/Datasets/Combined_BRCA/dataset-split/train_ids.txt', dtype=str)
    val_ids = np.loadtxt('/home/15t/Gul/Datasets/Combined_BRCA/dataset-split/val_ids.txt', dtype=str)
    test_ids = np.loadtxt('/home/15t/Gul/Datasets/Combined_BRCA/dataset-split/test_ids.txt', dtype=str)
    
    # Create a dictionary to map image IDs to their corresponding set
    id_to_set = {}
    for image_id in train_ids:
        id_to_set[image_id] = 'train'
    for image_id in val_ids:
        id_to_set[image_id] = 'val'
    for image_id in test_ids:
        id_to_set[image_id] = 'test'
    
    # Process only the 'train' set as in your original code
    for data_set in ['train']:
        # Use the anonymized header
        temp_csv = [df_all_info[0]]
        
        # For each line in the original data
        for i in range(1, len(all_info)):
            original_path = all_info[i, 0]
            image_id = original_path.split("/")[-1].split(".")[0]  # Extract the image ID from the file path
            
            if id_to_set.get(image_id, None) == data_set:
                # Add the corresponding anonymized row
                temp_csv.append(df_all_info[i])
        
        # Save the anonymized train dataset
        pd.DataFrame(temp_csv).to_csv(
            f"/home/15t/Gul/SG/sheeraz/result_archive/risk_elem/{data_dir}/{client}/DBLP-Scholar/325/{data_set}.csv",
            header=None,
            index=None,
        )
        
        # Also save the original train dataset as in your code
       # original_temp_csv = [all_info[0]]
       # for i in range(1, len(all_info)):
       #     image_id = all_info[i, 0].split("/")[-1].split(".")[0]
       #     if id_to_set.get(image_id, None) == data_set:
       #         original_temp_csv.append(all_info[i])
                
       # pd.DataFrame(original_temp_csv).to_csv(
       #     f"/home/15t/Gul/SG/sheeraz/result_archive/risk_elem/{data_dir}/{client}/DBLP-Scholar/325/{data_set}.csv",
       #     header=None,
       #     index=None,
       # )
    
    '''
    '''
    all_info = pd.read_csv(csv_path_list[0], header=None).to_numpy()[:, :2]

    for csv_path in csv_path_list:
        print(csv_path)
        csv = pd.read_csv(csv_path, header=None).to_numpy()[:, -1:]
        all_info = np.hstack((all_info, csv))

    pd.DataFrame(all_info).to_csv(
        "/home/Gul/SG/sheeraz/result_archive/risk_elem/{}/{}/DBLP-Scholar/pair_info_more.csv".format(data_dir, client), #change path to the save distribution
        header=None,
        index=None,
    )

    all_info = all_info.tolist()
    
    # Load the image IDs for each set
    train_ids = np.loadtxt('/home/Gul/Datasets/BCNB/BCNB/dataset-splitting/train_id.txt', dtype=str)
    val_ids = np.loadtxt('/home/Gul/Datasets/BCNB/BCNB/dataset-splitting/val_id.txt', dtype=str)
    test_ids = np.loadtxt('/home/Gul/Datasets/BCNB/BCNB/dataset-splitting/test_id.txt', dtype=str)
    
    # Updated Sheeraz 2/5/2024
    # Create a dictionary to map image IDs to their corresponding set
    id_to_set = {}
    for image_id in train_ids:
        id_to_set[image_id] = 'train'
    for image_id in val_ids:
        id_to_set[image_id] = 'val'
    for image_id in test_ids:
        id_to_set[image_id] = 'test'
    
    for data_set in ['train']:
        temp_csv = [all_info[0]]
        for line in all_info[1:]:
            image_id = line[0].split("/")[-1].split(".")[0]  # Extract the image ID from the file path
            if id_to_set.get(image_id, None) == data_set:
                temp_csv.append(line)
    
        pd.DataFrame(temp_csv).to_csv(
            "/home/Gul/SG/sheeraz/result_archive/risk_elem/{}/{}/DBLP-Scholar/325/{}.csv".format( #change path to the save distribution
                data_dir, client, data_set
            ),
            header=None,
            index=None,
        )
    '''
    '''
    # Load the image IDs for each set
    train_ids = np.loadtxt('/home/Gul/Datasets/Combined_BRCA/dataset-split/train_ids.txt', dtype=str)
    val_ids = np.loadtxt('/home/Gul/Datasets/Combined_BRCA/dataset-split/val_ids.txt', dtype=str)
    test_ids = np.loadtxt('/home/Gul/Datasets/Combined_BRCA/dataset-split/test_ids.txt', dtype=str)
    
    # Updated Sheeraz 2/5/2024
    # Create a dictionary to map image IDs to their corresponding set
    id_to_set = {}
    for image_id in train_ids:
        id_to_set[image_id] = 'train'
    for image_id in val_ids:
        id_to_set[image_id] = 'val'
    for image_id in test_ids:
        id_to_set[image_id] = 'test'
    
    for data_set in ['train']:
        temp_csv = [all_info[0]]
        for line in all_info[1:]:
            image_id = line[0].split("/")[-1].split(".")[0]  # Extract the image ID from the file path
            if id_to_set.get(image_id, None) == data_set:
                temp_csv.append(line)
    
        pd.DataFrame(temp_csv).to_csv(
            "/home/Gul/SG/sheeraz/result_archive/risk_elem/{}/{}/DBLP-Scholar/325/{}.csv".format( #change path to the save distribution
                data_dir, client, data_set
            ),
            header=None,
            index=None,
        )
    '''
    # for data_set in ['train', 'val', 'test']:
    #for data_set in ["train"]:
    #    temp_csv = [all_info[0]]

    #    for line in all_info[1:]:
    #        #print(line[0].split("/")[1])
    #        if line[0].split("/")[1] == data_set:
    #            temp_csv.append(line)

    #    pd.DataFrame(temp_csv).to_csv(
    #        "/home/ssd0/SG/sheeraz/result_archive/risk_elem/{}/DBLP-Scholar/325/{}.csv".format( #change path to the save distribution
    #            data_dir, data_set
    #        ),
    #        header=None,
    #        index=None,
    #    )


if __name__ == "__main__":

    get_one_risk_info()
