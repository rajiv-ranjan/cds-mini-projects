import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
                                    TimeDistributed, LSTM, GRU, BatchNormalization)
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import glob
import random
from sklearn.model_selection import train_test_split

# Feature extraction models
def build_custom_convnet(input_shape=(224, 224, 3)):
    """
    Build a custom ConvNet for feature extraction from individual frames.
    
    Parameters:
    input_shape (tuple): Input shape of a single frame (height, width, channels)
    
    Returns:
    model: Keras model for feature extraction
    """
    inputs = Input(shape=input_shape)
    
    # First convolutional block
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Second convolutional block
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Third convolutional block
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Fourth convolutional block
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Feature extraction part
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=x, name='custom_convnet')
    return model

def pretrained_feature_extractor(model_type='mobilenet', input_shape=(224, 224, 3), trainable=False):
    """
    Create a pre-trained model for feature extraction.
    
    Parameters:
    model_type (str): 'mobilenet' or 'resnet'
    input_shape (tuple): Input shape of a single frame (height, width, channels)
    trainable (bool): Whether to fine-tune the pre-trained model
    
    Returns:
    model: Keras model for feature extraction
    """
    if model_type == 'mobilenet':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_type == 'resnet':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError("model_type must be 'mobilenet' or 'resnet'")
    
    # Set the base model to be trainable or not
    base_model.trainable = trainable
    
    # Add custom layers on top of the pre-trained model
    inputs = Input(shape=input_shape)
    x = base_model(inputs)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=x, name=f'{model_type}_feature_extractor')
    return model

# Data generator class
class VideoFrameGenerator:
    def __init__(self, data_path, batch_size=8, num_frames=16, frame_height=224, frame_width=224, 
                 num_classes=6, shuffle=True, validation_split=0.2, seed=42):
        """
        Initialize a data generator for video frames.
        
        Parameters:
        data_path (str): Path to the data directory (train or test)
        batch_size (int): Batch size for training
        num_frames (int): Number of frames to use from each video
        frame_height, frame_width (int): Dimensions to resize frames to
        num_classes (int): Number of action classes
        shuffle (bool): Whether to shuffle data
        validation_split (float): Fraction of data to use for validation (only used for training data)
        seed (int): Random seed for reproducibility
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.validation_split = validation_split
        self.seed = seed
        
        # Get class names (folder names)
        self.class_names = sorted([d for d in os.listdir(data_path) 
                                  if os.path.isdir(os.path.join(data_path, d))])
        self.class_indices = {cls_name: i for i, cls_name in enumerate(self.class_names)}
        
        print(f"Found {len(self.class_names)} classes: {self.class_names}")
        
        # Collect all video folders
        self.video_paths = []
        self.video_labels = []
        
        for class_name in self.class_names:
            class_dir = os.path.join(data_path, class_name)
            video_folders = [d for d in os.listdir(class_dir) 
                            if os.path.isdir(os.path.join(class_dir, d))]
            
            for video_folder in video_folders:
                video_path = os.path.join(class_dir, video_folder)
                self.video_paths.append(video_path)
                self.video_labels.append(self.class_indices[class_name])
        
        # Split into train and validation if needed
        if validation_split > 0:
            random.seed(seed)
            indices = list(range(len(self.video_paths)))
            random.shuffle(indices)
            split_idx = int(len(indices) * (1 - validation_split))
            
            self.train_indices = indices[:split_idx]
            self.val_indices = indices[split_idx:]
            
            print(f"Total videos: {len(self.video_paths)}")
            print(f"Training videos: {len(self.train_indices)}")
            print(f"Validation videos: {len(self.val_indices)}")
        else:
            self.train_indices = list(range(len(self.video_paths)))
            self.val_indices = []
            
    def preprocess_frame(self, frame_path):
        """
        Load and preprocess a single frame.
        
        Parameters:
        frame_path (str): Path to the frame image
        
        Returns:
        array: Preprocessed frame as numpy array
        """
        # Load image
        img = load_img(frame_path, target_size=(self.frame_height, self.frame_width))
        img_array = img_to_array(img)
        
        # Normalize pixel values to [0, 1]
        img_array = img_array / 255.0
        
        return img_array
        
    def get_frames_from_video_folder(self, video_path):
        """
        Get sorted frames from a video folder.
        
        Parameters:
        video_path (str): Path to the video folder containing frame images
        
        Returns:
        list: List of frame file paths
        """
        # Get all frame files (.jpg, .jpeg, .png)
        frame_files = glob.glob(os.path.join(video_path, "*.jpg")) + \
                     glob.glob(os.path.join(video_path, "*.jpeg")) + \
                     glob.glob(os.path.join(video_path, "*.png"))
        
        # Sort frames by name (assuming sequential numbering)
        frame_files.sort()
        
        return frame_files
    
    def get_video_frames(self, video_path):
        """
        Load and preprocess frames from a video folder.
        
        Parameters:
        video_path (str): Path to the video folder
        
        Returns:
        array: Array of preprocessed frames
        """
        frame_files = self.get_frames_from_video_folder(video_path)
        
        # Handle case where we have fewer frames than needed
        if len(frame_files) < self.num_frames:
            # If there are not enough frames, duplicate the last frame
            frame_files = frame_files + [frame_files[-1]] * (self.num_frames - len(frame_files))
        
        # If there are more frames than needed, select evenly spaced frames
        if len(frame_files) > self.num_frames:
            # Calculate sampling interval
            interval = len(frame_files) / self.num_frames
            # Select evenly spaced frame indices
            selected_indices = [int(i * interval) for i in range(self.num_frames)]
            # Make sure we don't exceed the list bounds
            selected_indices = [min(idx, len(frame_files) - 1) for idx in selected_indices]
            # Get selected frames
            frame_files = [frame_files[i] for i in selected_indices]
        
        # Load and preprocess frames
        frames = np.array([self.preprocess_frame(f) for f in frame_files])
        
        return frames
    
    def generator(self, is_training=True):
        """
        Create a generator that yields batches of data.
        
        Parameters:
        is_training (bool): Whether to use training indices or validation indices
        
        Returns:
        generator: A generator yielding (batch_x, batch_y) tuples
        """
        indices = self.train_indices if is_training else self.val_indices
        
        # Create an array for the entire epoch
        if self.shuffle:
            random.shuffle(indices)
        
        # Calculate number of batches
        num_samples = len(indices)
        num_batches = int(np.ceil(num_samples / self.batch_size))
        
        # Generate batches
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # Initialize batch arrays
            batch_size = len(batch_indices)
            batch_x = np.zeros((batch_size, self.num_frames, self.frame_height, self.frame_width, 3))
            batch_y = np.zeros((batch_size, self.num_classes))
            
            for i, idx in enumerate(batch_indices):
                # Get video path and label
                video_path = self.video_paths[idx]
                label = self.video_labels[idx]
                
                # Get frames from video
                frames = self.get_video_frames(video_path)
                
                # Store in batch arrays
                batch_x[i] = frames
                batch_y[i] = to_categorical(label, self.num_classes)
            
            yield batch_x, batch_y
    
    def get_train_generator(self):
        """Get generator for training data."""
        return self.generator(is_training=True)
    
    def get_validation_generator(self):
        """Get generator for validation data."""
        return self.generator(is_training=False)
    
    def get_steps_per_epoch(self, is_training=True):
        """Get number of steps (batches) per epoch."""
        indices = self.train_indices if is_training else self.val_indices
        return int(np.ceil(len(indices) / self.batch_size))

# Model building function
def build_video_classifier(num_frames, frame_height, frame_width, channels, num_classes, 
                          use_pretrained=False, rnn_type='lstm', rnn_units=256):
    """
    Build a video classifier model with ConvNet + RNN + Dense architecture.
    
    Parameters:
    num_frames (int): Number of frames in each video sequence
    frame_height (int): Height of each frame
    frame_width (int): Width of each frame
    channels (int): Number of color channels (e.g., 3 for RGB)
    num_classes (int): Number of output classes
    use_pretrained (bool or str): False for custom ConvNet, or 'mobilenet'/'resnet' for pre-trained model
    rnn_type (str): 'lstm' or 'gru'
    rnn_units (int): Number of units in the RNN layer
    
    Returns:
    model: Compiled Keras model
    """
    # Define input shape for a sequence of frames
    input_shape = (num_frames, frame_height, frame_width, channels)
    
    # Select feature extractor
    if not use_pretrained:
        frame_feature_extractor = build_custom_convnet((frame_height, frame_width, channels))
    else:
        frame_feature_extractor = pretrained_feature_extractor(
            model_type=use_pretrained, 
            input_shape=(frame_height, frame_width, channels),
            trainable=False  # Set to True for fine-tuning
        )
    
    # Define the input layer for sequences
    sequence_input = Input(shape=input_shape)
    
    # Apply TimeDistributed wrapper to process each frame independently
    encoded_frames = TimeDistributed(frame_feature_extractor)(sequence_input)
    
    # Add recurrent layer to model temporal dependencies
    if rnn_type.lower() == 'lstm':
        x = LSTM(rnn_units, return_sequences=False)(encoded_frames)
    elif rnn_type.lower() == 'gru':
        x = GRU(rnn_units, return_sequences=False)(encoded_frames)
    else:
        raise ValueError("rnn_type must be 'lstm' or 'gru'")
    
    # Add classification layers
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create and compile model
    model = Model(inputs=sequence_input, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Training function
def train_model(model, train_gen, val_gen, train_steps, val_steps, 
               epochs=50, checkpoint_path='best_model.h5'):
    """
    Train the video classifier model with appropriate callbacks.
    
    Parameters:
    model: Compiled Keras model
    train_gen: Generator for training data
    val_gen: Generator for validation data
    train_steps: Number of steps (batches) per training epoch
    val_steps: Number of steps (batches) per validation epoch
    epochs (int): Maximum number of epochs to train for
    checkpoint_path (str): Path to save the best model
    
    Returns:
    history: Training history
    """
    # Define callbacks
    callbacks = [
        ModelCheckpoint(
            checkpoint_path, 
            monitor='val_accuracy', 
            save_best_only=True, 
            mode='max', 
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy', 
            patience=10, 
            restore_best_weights=True, 
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=5, 
            min_lr=1e-6, 
            verbose=1
        )
    ]
    
    # Train model
    history = model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=epochs,
        callbacks=callbacks
    )
    
    return history

# Utility to plot training history
def plot_training_history(history):
    """
    Plot training and validation accuracy/loss.
    
    Parameters:
    history: Training history returned by model.fit()
    """
    # Plot training & validation accuracy
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.show()

# Main execution for model training
def main():
    # Define parameters
    DATA_PATH_TRAIN = "preprocessed_videos/train"
    DATA_PATH_TEST = "preprocessed_videos/test"
    BATCH_SIZE = 8
    NUM_FRAMES = 16
    FRAME_HEIGHT = 224
    FRAME_WIDTH = 224
    NUM_CLASSES = 6
    EPOCHS = 50
    
    # Feature extraction and RNN configuration
    USE_PRETRAINED = 'mobilenet'  # 'mobilenet', 'resnet', or False for custom CNN
    RNN_TYPE = 'lstm'            # 'lstm' or 'gru'
    RNN_UNITS = 256
    
    # Initialize data generators
    train_data_gen = VideoFrameGenerator(
        data_path=DATA_PATH_TRAIN,
        batch_size=BATCH_SIZE,
        num_frames=NUM_FRAMES,
        frame_height=FRAME_HEIGHT,
        frame_width=FRAME_WIDTH,
        num_classes=NUM_CLASSES,
        shuffle=True,
        validation_split=0.2,
        seed=42
    )
    
    test_data_gen = VideoFrameGenerator(
        data_path=DATA_PATH_TEST,
        batch_size=BATCH_SIZE,
        num_frames=NUM_FRAMES,
        frame_height=FRAME_HEIGHT,
        frame_width=FRAME_WIDTH,
        num_classes=NUM_CLASSES,
        shuffle=False,
        validation_split=0,  # No validation split for test data
        seed=42
    )
    
    # Create generators
    train_gen = train_data_gen.get_train_generator()
    val_gen = train_data_gen.get_validation_generator()
    test_gen = test_data_gen.get_train_generator()  # Use train generator for all test data
    
    # Get steps per epoch
    train_steps = train_data_gen.get_steps_per_epoch(is_training=True)
    val_steps = train_data_gen.get_steps_per_epoch(is_training=False)
    test_steps = test_data_gen.get_steps_per_epoch(is_training=True)
    
    # Build the model
    model = build_video_classifier(
        num_frames=NUM_FRAMES,
        frame_height=FRAME_HEIGHT,
        frame_width=FRAME_WIDTH,
        channels=3,
        num_classes=NUM_CLASSES,
        use_pretrained=USE_PRETRAINED,
        rnn_type=RNN_TYPE,
        rnn_units=RNN_UNITS
    )
    
    # Print model summary
    model.summary()
    
    # Train the model
    history = train_model(
        model=model,
        train_gen=train_gen,
        val_gen=val_gen,
        train_steps=train_steps,
        val_steps=val_steps,
        epochs=EPOCHS,
        checkpoint_path='best_video_classifier.h5'
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = model.evaluate(test_gen, steps=test_steps)
    print(f"Test Loss: {test_results[0]:.4f}")
    print(f"Test Accuracy: {test_results[1]:.4f}")

if __name__ == "__main__":
    main()
