# dataset_splitter.py
import numpy as np
import torch
from torch.utils.data import random_split

def split_train_dataset_to_agents_server(K, input_trainset, BatchSize, data_num_per_user, data_num_at_server,
                                         cross_split, USE_NON_IID=1, global_data_fraction=0.01, global_val_fraction=0.2):
    """
    将训练集划分给各个客户端，并将全局数据集拆分为训练集和验证集。
    """
    trainset_all_users = [() for k in range(K + 1)]
    trainloader_all_users = [() for k in range(K + 1)]
    trainloader_all_users_train = [() for k in range(K)]
    trainloader_all_users_cross = [() for k in range(K)]
    full_length = len(input_trainset)

    global_data_count = int(global_data_fraction * full_length)

    # 全局共享数据集划分
    labels = np.array([y for _, y in input_trainset])
    unique_labels = np.unique(labels)
    label_indices = {label: np.where(labels == label)[0] for label in unique_labels}

    # 分配全局共享IID数据集
    global_indices = []
    per_class_count = global_data_count // len(unique_labels)
    for label in unique_labels:
        class_indices = label_indices[label]
        global_indices.extend(class_indices[:per_class_count])

    global_dataset = torch.utils.data.Subset(input_trainset, global_indices)

    # 划分全局数据集为训练集和验证集
    global_train_size = int((1 - global_val_fraction) * len(global_dataset))
    global_val_size = len(global_dataset) - global_train_size
    global_train_dataset, global_val_dataset = random_split(global_dataset, [global_train_size, global_val_size])

    global_train_loader = torch.utils.data.DataLoader(global_train_dataset, batch_size=BatchSize, shuffle=True, num_workers=0)
    global_val_loader = torch.utils.data.DataLoader(global_val_dataset, batch_size=BatchSize, shuffle=False, num_workers=0)

    remaining_indices = np.setdiff1d(np.arange(full_length), global_indices)

    # 根据 USE_NON_IID 全局变量决定是否使用Non-IID划分
    if USE_NON_IID == 1:
        # 使用 Non-IID 划分
        labels = np.array([y for _, y in input_trainset])
        label_indices = {label: np.where(labels == label)[0] for label in unique_labels}

        for k in range(K):
            selected_labels = np.random.choice(unique_labels, size=int(len(unique_labels) * 0.3), replace=False)
            selected_indices = []

            for label in selected_labels:
                if len(label_indices[label]) > 0:
                    # 选择样本，确保不会超过可用样本数量
                    available_indices = label_indices[label]
                    count_per_label = min(data_num_per_user // len(selected_labels), len(available_indices))
                    selected_indices.extend(available_indices[:count_per_label])

            np.random.shuffle(selected_indices)

            # 确保每个客户端的样本数量不超过 data_num_per_user
            selected_indices = selected_indices[:data_num_per_user]

            trainset_all_users[k] = torch.utils.data.Subset(input_trainset, selected_indices)
            trainloader_all_users[k] = torch.utils.data.DataLoader(trainset_all_users[k], batch_size=BatchSize,
                                                                   shuffle=True, num_workers=0)
            trainloader_all_users_train[k] = trainloader_all_users[k]
            trainloader_all_users_cross[k] = trainloader_all_users[k]

    else:
        # 使用 IID 划分
        for k in range(K):
            indices = np.random.choice(remaining_indices, data_num_per_user, replace=False)
            remaining_indices = np.setdiff1d(remaining_indices, indices)
            trainset_all_users[k] = torch.utils.data.Subset(input_trainset, indices)
            trainloader_all_users[k] = torch.utils.data.DataLoader(trainset_all_users[k], batch_size=BatchSize,
                                                                   shuffle=True, num_workers=0)
            trainloader_all_users_train[k] = trainloader_all_users[k]
            trainloader_all_users_cross[k] = trainloader_all_users[k]

    # 分配剩余的数据用于服务器端
    server_indices = np.setdiff1d(remaining_indices, np.concatenate([trainset_all_users[k].indices for k in range(K)]))
    trainset_all_users[K] = torch.utils.data.Subset(input_trainset, server_indices)
    trainloader_all_users[K] = torch.utils.data.DataLoader(trainset_all_users[K], batch_size=BatchSize,
                                                           shuffle=True, num_workers=0)

    return trainloader_all_users, trainset_all_users, trainloader_all_users_train, trainloader_all_users_cross, global_train_loader, global_val_loader
