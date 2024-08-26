import numpy as np
import time
import os
from models import *
from utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

(x_train, y_train), (x_test, y_test), end, input_shape = load_data('cifar10')

def train_model(worker_model, dataset, results, worker_id, batch_size):

    start = time.time()
    
    history = worker_model.fit(dataset[0], dataset[1], batch_size=batch_size, epochs=1, validation_data=(x_test, y_test), verbose=0)    
    loss = history.history['loss'][0]

    end = time.time()
    training_time = end - start
    
    results.append((worker_id, training_time, worker_model, loss))

num_workers = 16
batch_size = 64
epochs = 25
beta = [0.2, 0.4, 0.6, 0.8, 1.0]

for workers in range(4, 9):
    
    for portion in beta:

        sample_size = int(len(x_train)*portion)
        print(f"For Number of Workers = {workers} and Batch Size = {portion*100}% ({sample_size})")
        indices = np.random.choice(len(x_train), sample_size, replace=False)
        x_subsets, y_subsets = dirichlet_split((x_train[indices], y_train[indices]), num_workers)

        model = get_model(resnet18, input_shape, end)
        model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

        loss = []

        for global_epoch in range(epochs):

            start_time = time.time()
            results = []    
            models = []
            
            for i in range(workers):
                train_model(model, (x_subsets[i], y_subsets[i]), results, i, batch_size)

            # results.sort(key=lambda x: x[1])
            # results = results[:workers]
            
            models = [result[2] for result in results]
            logs = [result[3] for result in results]
            model = average_model_weights(models, 'resnet18', end, input_shape)    
            model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
            loss.append(np.mean(logs))
            end_time = time.time()
            total_time_taken = end_time - start_time
            print(f"Global Epoch {global_epoch + 1}/{epochs}: Loss={np.mean(logs)}, Time Taken={total_time_taken}")

        print("Training complete.")

        # avg_loss = []

        # for worker_id in range(workers):
        #     worker_losses = [metrics[epoch][worker_id] for epoch in range(epochs)]
        #     avg_loss.append(worker_losses)

        # avg_loss = np.mean(avg_loss, axis=0)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epochs+1), loss, label=f'Workers = {workers}, Beta = {portion}')

    s = len(indices)
    b_min = np.ceil((workers) * s / (workers + 1)) / s
    beta = [size for size in beta if size >= b_min]

plt.title('Training Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(range(1, epochs+1))
plt.grid(True, which='both', axis='both')

plt.legend()
plt.show()
