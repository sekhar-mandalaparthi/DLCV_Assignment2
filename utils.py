import torch
import matplotlib.pyplot as plt

# Plot learning curves of a model
def plot_learning_curves(results):
    """Plots training curves of a results dictionary."""
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(12, 4))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

def plot_learning_curves_vs_hparam(results_vs_hparam):
    """Plots learning curves vs different hyperparameter value."""
    hparams = results_vs_hparam.keys()
    epochs = range(len(results_vs_hparam[list(hparams)[0]]["train_loss"]))
    
    fig = plt.figure(figsize=(12, 8))
    for hparam in hparams:
        # Plot training loss
        plt.subplot(2, 2, 1)
        plt.plot(epochs, results_vs_hparam[hparam]["train_loss"], label=hparam)
        plt.title("Training Loss")
        plt.xlabel("Epochs")
        plt.legend()

        # Plot testing loss
        plt.subplot(2, 2, 2)
        plt.plot(epochs, results_vs_hparam[hparam]["test_loss"], label=hparam)
        plt.title("Test Loss")
        plt.xlabel("Epochs")
        plt.legend()

        # Plot training accuracy
        plt.subplot(2, 2, 3)
        plt.plot(epochs, results_vs_hparam[hparam]["train_acc"], label=hparam)
        plt.title("Training Accuracy")
        plt.xlabel("Epochs")
        plt.legend()

        # Plot testing accuracy
        plt.subplot(2, 2, 4)
        plt.plot(epochs, results_vs_hparam[hparam]["test_acc"], label=hparam)
        plt.title("Test Accuracy")
        plt.xlabel("Epochs")
        plt.legend()
    fig.subplots_adjust(hspace=0.3)
    plt.show()

def plot_predictions_vs_layers(samples, predictions_vs_layers):
    '''Plots the predictions of a model using the CLS token from each layer of the transformer.'''
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    rows = len(predictions_vs_layers.keys())+1
    cols = len(predictions_vs_layers[0])+1
    #print(f'rows: {rows}, cols: {cols}')
    fig = plt.figure(figsize=(cols, rows))
    fig.suptitle('Predictions using CLS token from each layer', fontsize=14)
    for l in range(cols-1):
        plt.subplot(rows, cols, l+2)
        plt.text(0.5, 0.15, f'Layer {l+1}', fontsize=12, ha='center')
        plt.axis('off')
    for i in range(rows-1):
        img = samples[i][0].squeeze().permute(1, 2, 0)
        gt = samples[i][1]
        preds = predictions_vs_layers[i]
        plt.subplot(rows, cols, (i+1) * cols + 1)
        plt.imshow(img)
        plt.xlabel(f'{classes[gt]}')
        plt.xticks([])
        plt.yticks([])
        for l in range(cols-1):
            color = 'green' if gt == preds[l] else 'red'
            plt.subplot(rows, cols, (i+1) * cols + l + 2)
            plt.text(0.5, 0.5, str(classes[preds[l]]), fontsize=14, ha='center', color=color)
            plt.axis('off')
            #plt.title(f'Layer {l}')
            plt.xlabel(f'Layer {l}')
            
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.5)
    plt.show()

def get_2_samples_per_class(dataset):
    '''Returns 2 random samples per class (total of 20) from the dataset.'''
    dataset_ = torch.utils.data.Subset(dataset, torch.randperm(len(dataset)))
    samples = []
    for i in range(10):
        for img, label in dataset_:
            if label == i:
                samples.append((img, label))
                if len(samples)%2 == 0:
                    break
        if len(samples) == 20:
            break
    return samples

def get_attentions(model, image):
    '''Returns the attention maps and the predicted label of an image.'''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    output = model(image.unsqueeze(dim=0).to(device))
    output_label = torch.argmax(output).item()
    attentions = []
    for i in range(len(model.transformer_encoder)):
        attentions.append(model.transformer_encoder[i].attn_weights)
    return attentions, output_label

def attention_rollout(attentions):
    # Initialize rollout with identity matrix
    rollout = torch.eye(attentions[0].size(-1)).to(attentions[0].device)

    # Multiply attention maps layer by layer
    for attention in attentions:
        attention_heads_fused = attention.mean(dim=1) # Average attention across heads
        attention_heads_fused += torch.eye(attention_heads_fused.size(-1)).to(attention_heads_fused.device) # A + I
        attention_heads_fused /= attention_heads_fused.sum(dim=-1, keepdim=True) # Normalizing A
        rollout = torch.matmul(rollout, attention_heads_fused) # Multiplication
    return rollout