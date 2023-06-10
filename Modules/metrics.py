import torchmetrics
from torchmetrics.functional import precision, recall, accuracy
from sklearn.metrics import confusion_matrix


def get_metrics(model, testloader):
    targets = torch.empty(0, dtype=torch.uint8)
    preds = torch.empty(0, dtype=torch.uint8)

    for data, labels in testloader:
        targets = torch.cat((targets, labels))

        data = data.to(device)
        outputs = model(data).cpu()
        _, pred = torch.max(outputs, 1)
        preds = torch.cat((preds, pred))

    prec = precision(preds, targets, task='multiclass', num_classes=3, multidim_average='global', average=None)
    rec = recall(preds, targets, task='multiclass', num_classes=3, multidim_average='global', average=None)

    classes = ["Normal", "Non-COVID Pneumonia", "COVID-19 Pneumonia"]

    for i in range(3):
        print("----------------------------\n")
        print("Class", i, ":", classes[i], '\n')
        print("Precision:", prec[i].item())
        print("Recall:", rec[i].item(), "\n")

    print("----------------------------\n")
    print("Overall Accuracy: ", accuracy(preds, targets, task='multiclass', num_classes=3).item(), "\n\n")

    conf_matrix = confusion_matrix(preds.numpy(), targets.numpy())

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Greens, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()
