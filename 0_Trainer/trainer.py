import tensorflow as tf
import time
from tqdm import tqdm
import numpy as np

class Trainer:
    def __init__(self, loss_fn, metrics, model, train_dataloader, val_dataloader, optimizer, cps_dir, logs_dir, test_name, best_loss_threshold=None):
        """
        loss_fn: loss function
        metrics: metrics to be calculated (not implemented yet)
        model: model
        train_dataloader: train dataloader
        val_dataloader: validation dataloader
        optimizer: optimizer
        cps_dir: directory to save checkpoint
        logs_dir: directory to save logs for tensorboard
        test_name: test name
        best_loss_threshold: minimum validation loss for saving best val loss checkpoint
        """
        
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.cps_dir = cps_dir
        self.logs_dir = logs_dir
        self.test_name = test_name
        self.min_loss = best_loss_threshold if best_loss_threshold is not None else float('inf')

        # tmp. # To-do: make function to get metrics as argument and generate it automatically
        self.train_acc_metric  = tf.keras.metrics.Accuracy()
        self.val_acc_metric = tf.keras.metrics.Accuracy()
        

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            # forward pass
            probs = self.model(x, training=True)
            # compute loss value for the minibatch 
            train_loss = self.loss_fn(y, probs)
            
        # use the gradient tape to automatically retrive the gradients of the trainable variables with respect to the loss    
        grads = tape.gradient(train_loss, self.model.trainable_weights) 

        # run one step of gradient descent by updating the value of the variables to minimize the loss
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        
        # Update training metric
        y_pred = tf.reshape(tf.argmax(probs, axis=1), shape=(-1, 1))            
        self.train_acc_metric.update_state(y, y_pred)  
        
        return train_loss
    
    @tf.function
    def test_step(self, x, y):
        val_probs = self.model(x, training=False)

        # update val metrics
        y_pred = tf.reshape(tf.argmax(val_probs, axis=1), shape=(-1, 1))            
        self.val_acc_metric.update_state(y, y_pred)
        
        # validation loss
        val_loss = self.loss_fn(y, val_probs)
        return val_loss
    
    def train_one_epoch(self):
        
        n_steps = len(self.train_dataloader)
        enumerated_batches = tqdm(
            enumerate(self.train_dataloader),
            total = n_steps,
            unit="step",
            ascii = ' >=',
            bar_format="{n_fmt}/{total_fmt} [{bar:30}] {rate_fmt}{postfix}",
            position=0,
            leave=True)

        tot_tra_loss = []
        for (step, (x_batch_train, y_batch_train)) in enumerated_batches:
            step_start=time.time()
            batch_loss = self.train_step(x_batch_train, y_batch_train)
            tot_tra_loss.append(batch_loss)
            step_end=time.time()
            enumerated_batches.set_description(f"{step+1}/{n_steps}")
            enumerated_batches.set_postfix_str(
                s=f"loss: {float(batch_loss):.4f} | acc: {float(self.train_acc_metric.result()):.4f} | time: {step_end-step_start:.3f} s", refresh=True)

        avg_tra_loss = np.mean(tot_tra_loss)
        # get metric and reset state
        train_acc = self.train_acc_metric.result()
        self.train_acc_metric.reset_states()
        
        return avg_tra_loss, train_acc
    
    def val_one_epoch(self):
        tot_val_loss = []
        for (step, (x_batch_val, y_batch_val)) in enumerate(self.val_dataloader):
            batch_loss = self.test_step(x_batch_val, y_batch_val)
            tot_val_loss.append(batch_loss)

        avg_val_loss = np.mean(tot_val_loss)

        # get metric and reset state
        val_acc = self.val_acc_metric.result()
        self.val_acc_metric.reset_states()

        return avg_val_loss, val_acc
    
    def run(self, epochs):
        writer = tf.summary.create_file_writer(self.logs_dir)

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs} -- {self.test_name}")
            epoch_start = time.time()
            
            # train process
            train_start = time.time()
            train_loss, train_acc = self.train_one_epoch()
            train_end = time.time()
            
            print(f"# Train -- loss: {train_loss:.4f} | acc: {train_acc:.4f} | time: {train_end-train_start:.3f} s ({(train_end-train_start)/60:.3f} min)")

            # save checkpoint
            self.model.save_weights(f"{self.cps_dir}/checkpoint")

            # validation process
            val_start = time.time()
            val_loss, val_acc = self.val_one_epoch()
            val_end = time.time()
            print(f"# Val   -- loss: {val_loss:.4f} | acc: {val_acc:.4f} | time: {val_end-val_start:.3f} s ({(val_end-val_start)/60:.3f} min)\n")
                
            self.train_dataloader.on_epoch_end()
            self.val_dataloader.on_epoch_end()
            epoch_end = time.time()
            
            # save best metric
            if val_loss < self.min_loss:
                print(f"#### Epoch {epoch+1} -- Val loss improved from {self.min_loss:.4f} to {val_loss:.4f} | Time: {(epoch_end-epoch_start)/60:.3f} min ####")
                self.model.save_weights(f"{self.cps_dir}/BestValLoss")
                self.min_loss = val_loss
            else:
                print(f"#### Epoch {epoch+1} -- Val loss did not improve from {self.min_loss:.4f} | Time: {(epoch_end-epoch_start)/60:.3f} min ####")

            # logging for tensorboard
            with writer.as_default():
                tf.summary.scalar("train_loss", train_loss, step=epoch)
                tf.summary.scalar("train_acc", train_acc, step=epoch)
                tf.summary.scalar("val_loss", val_loss, step=epoch)
                tf.summary.scalar("val_acc", val_acc, step=epoch)
            writer.flush()
        return self.model
