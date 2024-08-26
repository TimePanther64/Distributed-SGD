import tensorflow as tf
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100  # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
import numpy as np
import matplotlib.pyplot as plt

def load_data(dataset='mnist'):

    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
        x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        end = 10
        input_shape = (28, 28, 1)
    
    elif dataset == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
        x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
        # y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        end = 10
        input_shape = (28, 28, 1)
    
    elif dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        # y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        end = 10
        input_shape = (32, 32, 3)

    elif dataset == 'cifar100':
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        # y_train = to_categorical(y_train, 100)
        y_test = to_categorical(y_test, 100)
        end = 100
        input_shape = (32, 32, 3)
    
    return (x_train, y_train), (x_test, y_test), end, input_shape

def dirichlet_split(train_set, num_threads=16, alpha=1.0):

    x_train, y_train = train_set
    labels = np.unique(y_train)
    label_x = [[] for _ in range(len(labels))]
    label_y = [[] for _ in range(len(labels))]
    label_dir = [[] for _ in range(len(labels))]
    x_subset = [[] for _ in range(num_threads)]
    y_subset = [[] for _ in range(num_threads)]

    for label in labels:
        indices = np.where(y_train == label)[0]
        label_x[label] = x_train[indices]
        label_y[label] = y_train[indices]
        label_dir[label] = np.argmax(np.random.dirichlet([alpha]*num_threads, size=len(indices)), axis=1)

        for i, w in enumerate(label_dir[label]):
            x_subset[w].append(label_x[label][i])
            y_subset[w].append(label_y[label][i])
        
    x_subsets = [np.array(x_subset[i]) for i in range(num_threads)]
    y_subsets = [to_categorical(np.array(y_subset[i]), 10) for i in range(num_threads)]

    return x_subsets, y_subsets

# class MetricsLogger(tf.keras.callbacks.Callback):
#     def on_train_begin(self, logs=None):
#         self.epoch = []
#         self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

#     def on_epoch_end(self, epoch, logs=None):
#         self.epoch.append(epoch)
#         self.history['loss'].append(logs.get('loss'))
#         self.history['accuracy'].append(logs.get('accuracy'))
#         self.history['val_loss'].append(logs.get('val_loss'))
#         self.history['val_accuracy'].append(logs.get('val_accuracy'))

# def plot_metrics(histories):
#     epochs = range(len(histories[0]['loss']))

#     for i, history in enumerate(histories):
#         plt.figure(figsize=(12, 4))

#         plt.subplot(1, 2, 1)
#         plt.plot(epochs, history['loss'], label='Training loss')
#         plt.plot(epochs, history['val_loss'], label='Validation loss')
#         plt.title(f'Model {i+1} - Training and Validation Loss')
#         plt.xlabel('Epochs')
#         plt.ylabel('Loss')
#         plt.legend()

#         plt.subplot(1, 2, 2)
#         plt.plot(epochs, history['accuracy'], label='Training accuracy')
#         plt.plot(epochs, history['val_accuracy'], label='Validation accuracy')
#         plt.title(f'Model {i+1} - Training and Validation Accuracy')
#         plt.xlabel('Epochs')
#         plt.ylabel('Accuracy')
#         plt.legend()

#         plt.show()

# def average_histories(histories, ge):

#     metric_label = ['loss', 'accuracy', 'val_loss', 'val_accuracy']
#     temp = [[[] for _ in range(len(histories))] for _ in range(len(metric_label))]
#     avg_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
#     params = []

#     for ir, rics in histories:
#         for size, wor, metrix in rics:
#             if [size, wor] not in params:
#                 params.append([size, wor])
#             for ep, met in metrix:
#                 for i, m in enumerate(metric_label):
#                     temp[i][ir].append(met[0][m])

#     for lab, t in zip(metric_label, temp):
#         avg_history[lab].append(np.reshape(np.ma.masked_equal(t, 0).mean(axis=0).compressed(), (len(params), ge)))

#     for m in metric_label:
#         avg_history[m] = np.hstack((params, np.squeeze(avg_history[m])))

#     return avg_history

# def plot_accuracies(accuracies):
#     plt.figure(figsize=(10, 6))
#     plt.plot(range(len(accuracies)), accuracies, marker='o', linestyle='-', color='b')
#     plt.title('Model Accuracy After Each Iteration Step')
#     plt.xlabel('Iteration Step')
#     plt.ylabel('Accuracy')
#     plt.ylim(0, 1) 
#     plt.grid(True)
#     plt.show()

# def preprocess(image, label):
#     image = tf.cast(image, tf.float32) / 255.0
#     label = tf.one_hot(label, depth=47)
#     return image, label

# def dataset_to_numpy(dataset):
#     images, labels = [], []
#     for image, label in dataset:
#         images.append(image.numpy())
#         labels.append(label.numpy())
#     return np.array(images), np.array(labels)


# def average_histories(histories, ge):

#     metric_label = ['loss', 'accuracy', 'val_loss', 'val_accuracy']
#     temp = [[[] for _ in range(len(histories))] for _ in range(len(metric_label))]
#     avg_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
#     params = []

#     for ir, rics in histories:
#         for size, wor, metrix in rics:
#             if [size, wor] not in params:
#                 params.append([size, wor])
#             for _, met in metrix:
#                 for i, m in enumerate(metric_label):
#                     temp[i][ir].append(met[0][m])

#     for lab, t in zip(metric_label, temp):
#         avg_history[lab].append(np.reshape(np.ma.masked_equal(t, 0).mean(axis=0).compressed(), (len(params), ge)))

#     for m in metric_label:
#         avg_history[m] = np.hstack((params, np.squeeze(avg_history[m])))

#     return avg_history
