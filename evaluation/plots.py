# Has to implement graphs like compression ratio vs accuracy and accuracy vs speedup
from matplotlib import pyplot as plt
import torch.nn as nn

from evaluation.evals import accuracy
from pruning.unstructured import *


def plot_compression_ratio_vs_accuracy(compression_ratios, accuracies):
    """Plot compression ratio vs accuracy

    Arguments:
        compression_ratios {iterable} -- List of compression ratios
        accuracies {iterable} -- List of accuracies
    """
    plt.plot(compression_ratios, accuracies)
    plt.xlabel("Compression ratio")
    plt.ylabel("Accuracy")
    plt.title("Compression ratio vs accuracy")
    plt.show()


def compare_compression_ratio_vs_accuracy(model, dataset):
    """Compare compression ratio vs accuracy for different models"""
    train_loader, test_loader = dataset.get_dataloader()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    compression_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    accuracies = []
    for compression_ratio in compression_ratios:
        trainer = UnstructuredL1normPrune(model, 5, train_loader, criterion, optimizer, compression_ratio)
        original_model, pruned_model = trainer.train_prune_retrain()
        accuracies.append(accuracy(pruned_model, test_loader))

    print("Accuracy of the model without pruning is " + str(accuracy(original_model, test_loader)))
    print("compression_ratios = " + str(compression_ratios))
    print("accuracies = " + str(accuracies))
    plot_compression_ratio_vs_accuracy(compression_ratios, accuracies)
