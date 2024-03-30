# Has to implement graphs like compression ratio vs accuracy and accuracy vs speedup
from matplotlib import pyplot as plt
import torch.nn as nn

from evaluation.evals import accuracy
from pruning.unstructured import *


# def plot_compression_ratio_vs_accuracy(compression_ratios, accuracies):
#     """Plot compression ratio vs accuracy

#     Arguments:
#         compression_ratios {iterable} -- List of compression ratios
#         accuracies {iterable} -- List of accuracies
#     """
#     plt.plot(compression_ratios, accuracies)
#     plt.xlabel("Compression ratio")
#     plt.ylabel("Accuracy")
#     plt.title("Compression ratio vs accuracy")
#     plt.show()


# def compare_compression_ratio_vs_accuracy(model, dataset , pruning_list):
#     """Compare compression ratio vs accuracy for different models"""
#     train_loader, test_loader = dataset.get_dataloader()
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     compression_ratios = [0.1, 0.2,0.3, 0.4,0.5,0.6, 0.7, 0.8,0.9,1.0]
#     for pruning in pruning_list: 
#         accuracies = [[]]
#         for compression_ratio in compression_ratios:
#             trainer = pruning(model, 5, train_loader, criterion, optimizer, compression_ratio)
#             pruned_model = trainer.prune_model()
#             accuracies.append(accuracy(pruned_model, test_loader))
        
#         # print("compression_ratios = " + str(compression_ratios))
#         # print("accuracies = " + str(accuracies))
#     plot_compression_ratio_vs_accuracy(compression_ratios, accuracies)

def compare_compression_ratio_vs_accuracy(model, dataset , pruning_list):
    """Compare compression ratio vs accuracy for different models"""
    train_loader, test_loader = dataset.get_dataloader()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    compression_ratios = [0.1, 0.2,0.3, 0.4,0.5,0.6, 0.7, 0.8,0.9,1.0]
    for name, pruning in pruning_list.items(): 
        accuracies = []
        for compression_ratio in compression_ratios:
            pruning.setargs(model, 5, train_loader, criterion, optimizer, compression_ratio)
            # pruning(model, 5, train_loader, criterion, optimizer, compression_ratio)
            pruned_model = pruning.prune_model()
            accuracies.append(accuracy(pruned_model, test_loader))
        
        plt.plot(compression_ratios, accuracies, label=name)

    plt.xlabel("Compression ratio")
    plt.ylabel("Accuracy")
    plt.title("Compression ratio vs accuracy")
    plt.legend()
    plt.show()
    
    # def compare_compression_ratio_vs_latency(model)