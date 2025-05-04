# -*- coding: utf-8 -*-

## 20240830 用惊蛰
## 用原始数据

import torch
import warnings

import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from torch.utils.tensorboard import SummaryWriter
import os
import time
from torch.cuda import amp
import datetime
from spikingjelly import visualizing
from torch.utils.data import DataLoader
from CustomCSVMatrixDataset import CustomCSVMatrixDataset
import matplotlib.pyplot as plt
import itertools
from pytsk.gradient_descent.antecedent import antecedent_init_p, ThresholdedATanMF
from pytsk.gradient_descent.tsk import TSK
import switchable_norm as sn

# 在layer中添加此类
# class SwitchNorm2d(sn.SwitchNorm2d, base.StepModule):
#     def __init__(
#             self,
#             num_features,
#             eps=1e-5,
#             momentum=0.1,
#             affine=True,
#             track_running_stats=True,
#             step_mode='s'
#     ):
#         """
#         * :ref:`API in English <SwitchNorm2d-en>`
#
#         .. _SwitchNorm2d-cn:
#
#         :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
#         :type step_mode: str
#
#         其他的参数API参见 :class:`sn.SwitchNorm2d`
#
#         * :ref:`中文 API <SwitchNorm2d-cn>`
#
#         .. _SwitchNorm2d-en:
#
#         :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
#         :type step_mode: str
#
#         Refer to :class:`sn.SwitchNorm2d` for other parameters' API
#         """
#         super().__init__(num_features, eps, momentum, affine, track_running_stats)
#         self.step_mode = step_mode
#
#     def extra_repr(self):
#         return super().extra_repr() + f', step_mode={self.step_mode}'
#
#     def forward(self, x: Tensor):
#         if self.step_mode == 's':
#             return super().forward(x)
#
#         elif self.step_mode == 'm':
#             if x.dim() != 5:
#                 raise ValueError(f'expected x with shape [T, N, C, H, W], but got x with shape {x.shape}!')
#             return functional.seq_to_ann_forward(x, super().forward)

window=50

warnings.filterwarnings('ignore', category=UserWarning, module='np._distributor_init')
Num_Class=3
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, save_path='confusion_matrix.png'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : matrix
    - classes
    - normalize : True:percent, False:number
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.figure(dpi=150)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2%' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path)
    plt.close()


class NeuralNetworks(nn.Module):
    def __init__(self, T: int, channels: int, n_rule: int, n_class: int, order: int, use_cupy=False, device='cpu'):
        super(NeuralNetworks, self).__init__()
        self.T = T
        self.device = device

        self.conv_fc = nn.Sequential(
            layer.Conv2d(1, channels, kernel_size=3, padding=1, bias=False),  # 50 * 36
            layer.SwitchNorm2d(channels),
            layer.Dropout2d(p=0.3),  # 添加Dropout层
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),  # 25 * 18

            layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            layer.Dropout2d(p=0.2),  # 添加Dropout层
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),  # 12 * 9

            layer.Flatten(),  # 展平层
            layer.Linear(channels * 12 * 9, 512, bias=False),
            layer.Dropout(p=0.5),  # 线性层之后，激活层之前的Dropout
            neuron.IFNode(surrogate_function=surrogate.ATan()),

        ).to(device)

        self.tsk = None  # 初始时TSK模型为空

        functional.set_step_mode(self, step_mode='m')

        if use_cupy:
            functional.set_backend(self, backend='cupy')

    def initialize_tsk(self, X, n_rule, n_class, order):
        """
        使用中间层输出初始化TSK模型。

        :param X: 中间层输出，形状为 (N, D)
        :param n_rule: 规则数量
        :param n_class: 输出类别数
        :param order: TSK模型的阶数
        """
        # 初始化隶属函数中心
        init_p=antecedent_init_p(X, n_rule)
        gmf = nn.Sequential(
            ThresholdedATanMF(in_dim=X.shape[1], n_rule=n_rule, high_dim=True, init_p=init_p, alpha=10),
            # sn.SwitchNorm1d(n_rule),
            nn.ReLU()
        ).to(self.device)
        # 初始化TSK模型
        self.tsk = TSK(
            in_dim=X.shape[1],
            out_dim=n_class,
            n_rule=n_rule,
            antecedent=gmf,
            order=order
        ).to(self.device)

    def forward(self, x: torch.Tensor):
        # x.shape = [N, C, H, W]
        # add channel
        x = x.unsqueeze(1)  # [N, 60, 36] -> [N, 1, 60, 36]
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]

        x_seq = self.conv_fc(x_seq)

        fr = x_seq.mean(0)  # [T, N, C*H*W] -> [N, C*H*W]

        fr=fr.to(self.device)

        # 通过TSK模型
        if self.tsk is not None:
            output = self.tsk(fr)
        else:
            raise RuntimeError("TSK model is not initialized. Please call initialize_tsk before training.")

        return output

    def spiking_encoder(self):
        return self.conv_fc[0:3]

def classification_accuracy_std(input: torch.Tensor, target: torch.Tensor) -> float:
    # 将 target 转换回类别标签
    labels = target.argmax(dim=1)
    # 获取预测结果
    _, predicted = torch.max(input.data, 1)

    # 计算每个类别的准确率
    accuracies = []
    num_classes = input.shape[1]
    for class_idx in range(num_classes):
        total_class = (target == class_idx).sum().item()
        if total_class == 0:
            # 如果某类没有样本，则准确率为NaN
            accuracies.append(float('nan'))
        else:
            correct_class = ((predicted == class_idx) & (labels == class_idx)).sum().item()
            accuracy = correct_class / total_class
            accuracies.append(accuracy)

    # 计算准确率的标准差
    accuracies = np.array(accuracies)
    std_dev = np.nanstd(accuracies)  # 忽略NaN值
    return std_dev

def main(T, train_path, train_label_path, test_path, test_label_path, device,
         batch_size, epochs, num_workers, out_dir, resume_path, use_amp, use_cupy,
         optimizer_type, momentum, learning_rate, channels, save_spikes_dir, n_rule, out_dim):

    train = CustomCSVMatrixDataset(
        data_path=train_path,
        labels_path=train_label_path,
        input_shape=(window, 36)
    )
    test = CustomCSVMatrixDataset(
        data_path=test_path,
        labels_path=test_label_path,
        input_shape=(window, 36),
        mean_values=train.mean_values,
        std_values=train.std_values
    )

    # Create DataLoaders
    train_data_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True)
    test_data_loader = DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True)

    net = NeuralNetworks(T=T, channels=channels, n_rule=n_rule, n_class=out_dim, order=1, use_cupy=use_cupy, device=device)
    print(net)
    net.to(device)
    ## 创建模型实例,只有NeuralNetworks采用
    # 获取中间层输出
    with torch.no_grad():
        all_features = []
        all_labels = []
        for img, label in train_data_loader:
            img = img.to(device)
            label = label.to(device)
            output = net.conv_fc(img.unsqueeze(1).unsqueeze(0).repeat(net.T, 1, 1, 1, 1)).mean(0)
            all_features.append(output.cpu().numpy())
            # all_labels.append(label.cpu().numpy())
            functional.reset_net(net)

        all_features = np.vstack(all_features)
        # all_labels = np.concatenate(all_labels)

    # 初始化TSK模型
    net.initialize_tsk(all_features, n_rule=n_rule, n_class=out_dim, order=1)
    ## NeuralNetworks采用结束

    scaler = None
    if use_amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    max_test_acc = -1

    optimizer = None
    if optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    elif optimizer_type == 'adam':
        # optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)
    else:
        raise NotImplementedError(optimizer_type)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    if resume_path:
        checkpoint = torch.load(resume_path, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']

        if save_spikes_dir is not None and save_spikes_dir != '':
            encoder = net.spiking_encoder()
            with torch.no_grad():
                for img, label in test_data_loader:
                    img = img.to(device)
                    label = label.to(device)
                    # img.shape = [N, C, H, W]
                    img_seq = img.unsqueeze(0).repeat(net.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]
                    spike_seq = encoder(img_seq)
                    functional.reset_net(encoder)
                    to_pil_img = torchvision.transforms.ToPILImage()
                    vs_dir = os.path.join(save_spikes_dir, 'visualization')
                    os.mkdir(vs_dir)

                    img = img.cpu()
                    spike_seq = spike_seq.cpu()

                    img = F.interpolate(img, scale_factor=4, mode='bilinear')
                    # 28 * 28 is too small to read. So, we interpolate it to a larger size

                    for i in range(label.shape[0]):
                        vs_dir_i = os.path.join(vs_dir, f'{i}')
                        os.mkdir(vs_dir_i)
                        to_pil_img(img[i]).save(os.path.join(vs_dir_i, f'input.png'))
                        for t in range(net.T):
                            print(f'saving {i}-th sample with t={t}...')
                            # spike_seq.shape = [T, N, C, H, W]

                            visualizing.plot_2d_feature_map(spike_seq[t][i], 8, spike_seq.shape[2] // 8, 2, f'$S[{t}]$')
                            plt.savefig(os.path.join(vs_dir_i, f's_{t}.png'), pad_inches=0.02)
                            plt.savefig(os.path.join(vs_dir_i, f's_{t}.pdf'), pad_inches=0.02)
                            plt.savefig(os.path.join(vs_dir_i, f's_{t}.svg'), pad_inches=0.02)
                            plt.clf()

                    exit()

    out_dir = os.path.join(out_dir, f'T{T}_b{batch_size}_{optimizer_type}_lr{learning_rate}_c{channels}_r{n_rule}')

    if use_amp:
        out_dir += '_amp'

    if use_cupy:
        out_dir += '_cupy'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')

    # 初始化日志记录器
    writer = SummaryWriter(out_dir, purge_step=start_epoch)

    # 写入参数信息
    params_info = (
        f"T: {T}\n"
        f"Train_path: {train_path}\n"
        f"Train_label_path: {train_label_path}\n"
        f"Test_path: {test_path}\n"
        f"Test_label_path: {test_label_path}\n"
        f"Device: {device}\n"
        f"Batch Size: {batch_size}\n"
        f"Epochs: {epochs}\n"
        f"Number of Workers: {num_workers}\n"
        f"Output Directory: {out_dir}\n"
        f"Resume Path: {resume_path}\n"
        f"Use AMP: {use_amp}\n"
        f"Use CuPy: {use_cupy}\n"
        f"Optimizer Type: {optimizer_type}\n"
        f"Momentum: {momentum}\n"
        f"Learning Rate: {learning_rate}\n"
        f"Channels: {channels}\n"
        f"Save Spikes Directory: {save_spikes_dir}"
    )

    # 将参数信息写入文件
    args_file_path = os.path.join(out_dir, 'args.txt')
    with open(args_file_path, 'w', encoding='utf-8') as args_txt:
        args_txt.write(params_info)

    # 训练
    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for img, label in train_data_loader:
            optimizer.zero_grad()
            img = img.to(device)
            label = label.to(device)
            label_onehot = F.one_hot(label, Num_Class).float()

            if scaler is not None:
                with amp.autocast():
                    out_fr = net(img)
                    loss = F.cross_entropy(out_fr, label_onehot)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out_fr = net(img)
                loss = F.cross_entropy(out_fr, label_onehot)
                loss.backward()
                optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(net)

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        lr_scheduler.step()

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for img, label in test_data_loader:
                img = img.to(device)
                label = label.to(device)
                label_onehot = F.one_hot(label, Num_Class).float()
                out_fr = net(img)
                loss = F.cross_entropy(out_fr, label_onehot)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                # 收集预测结果和真实标签
                all_preds.extend(out_fr.argmax(1).cpu().numpy())
                all_labels.extend(label.cpu().numpy())
                functional.reset_net(net)

        # 保存测试结果到文件
        results_df = pd.DataFrame({'True Label': all_labels, 'Predicted Label': all_preds})
        results_df.to_csv(os.path.join(out_dir,'test_results.csv'), index=False)

        # Compute confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        # Assuming you have class names defined somewhere
        class_names = ['Stable', 'Shaking', 'Slipping']  # Replace with actual class names

        # Plot the confusion matrix
        plot_confusion_matrix(cm, classes=class_names, normalize=True, title='Confusion Matrix', save_path=os.path.join(out_dir,'Confusion Matrix.png'))

        test_time = time.time()
        test_speed = test_samples / (test_time - train_time)
        test_loss /= test_samples
        test_acc /= test_samples
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))

        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))

        print(out_dir)
        print(
            f'epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
        print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        print(
            f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')


if __name__ == '__main__':
    T = 4

    # your dataset

    train_path = ''
    train_label_path = ''
    test_path = ''
    test_label_path = ''

    device = 'cuda:0'
    batch_size = 128
    epochs = 100
    num_workers = 2
    out_dir = './logs'
    resume_path = None
    use_amp = True
    use_cupy = True
    optimizer_type = 'adam'
    momentum = 0.9
    learning_rate = 0.001
    channels = 64
    save_spikes_dir = None
    n_rule = 15
    out_dim = 3

    main(T, train_path, train_label_path, test_path, test_label_path, device, batch_size, epochs, num_workers, out_dir, resume_path, use_amp, use_cupy, optimizer_type, momentum, learning_rate, channels, save_spikes_dir, n_rule, out_dim)

