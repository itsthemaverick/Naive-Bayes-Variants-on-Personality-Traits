import matplotlib.pyplot as plt

def plot_accuracy(metrics:dict):
    models = list(metrics.keys())
    accuracies = [metrics[m]["accuracy"] for m in models]

    plt.figure()
    plt.bar(models,accuracies)
    plt.xlabel("Naive Bayes Variant")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison of Naive Bayes Models")
    plt.tight_layout()
    plt.show()