# Has to implement graphs like compression ratio vs accuracy and accuracy vs speedup
from matplotlib import pyplot as plt
import torch.nn as nn

from evaluation.evals import accuracy, measure_latency, measure_speedup
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
    compression_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    model_latency = measure_latency(model, test_loader)
    # print("Model latency = " + str(model_latency))
    pruning_technique_vs_accuracy = {}
    pruning_technique_vs_speedup = {}
    for name, pruning in pruning_list.items(): 
        accuracies = []
        speedups = []
        for compression_ratio in compression_ratios:
            pruning.setargs(model, 5, train_loader, criterion, optimizer, compression_ratio)
            # pruning(model, 5, train_loader, criterion, optimizer, compression_ratio)
            pruned_model = pruning.prune_model()
            accuracies.append(accuracy(pruned_model, test_loader))
            pruned_model_latency = measure_latency(pruned_model, test_loader)
            # print(name+" "+"Pruned model latency = " + str(pruned_model_latency))
            # speedups.append(model_latency/pruned_model_latency)
            speedups.append(measure_speedup(model, pruned_model, test_loader))
        
        # plt.plot(compression_ratios, accuracies, label=name)
        pruning_technique_vs_accuracy[name] = accuracies
        pruning_technique_vs_speedup[name] = speedups
        # print()
        

    for name, accuracies in pruning_technique_vs_accuracy.items():
        plt.plot(compression_ratios, accuracies, label=name)

    plt.xlabel("Compression ratio")
    plt.ylabel("Accuracy")
    plt.title("Compression ratio vs accuracy")
    plt.legend()
    plt.show()

    for name, speedups in pruning_technique_vs_speedup.items():
        plt.plot(compression_ratios, speedups, label=name)

    plt.xlabel("Compression ratio")
    plt.ylabel("Speedup")
    plt.title("Compression ratio vs Speedup")
    plt.legend()
    plt.show()
    
    # def compare_compression_ratio_vs_latency(model)