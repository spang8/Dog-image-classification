# Dog-image-classification
Use convolutional neural network to to dog images and classify dog breed

## Project procedure overview
* Load data
    * Get all image path: np.array(glob())  
* Use `pre-trained model` to detect `human faces`
    * Load pre-trained model: cv2.CascadeClassifier
    * read image file: cv2.imread('path'): BGR
    * convert to gray scale  
    * Use pre-trained model to detect faces 
    * square the deteced face 
    * convert BGR to RGB, and show image  
    * Test how the `human face detector` detect `human faces` and `dog faces`
* Use `pre-trained model` to detect `dogs`  
    * Load pre-trained model: torchvision.models.vgg16 
    * Directly open image (Image.open())  
    * resize image and ToTensor  
    * predict dog breed using vgg16 pre-trained model (output probability of 1000 animals, need to get the max prob as the predicted class)  
    * among 1000 animals in vgg16 model, 151-268 are dog breeds. Only detect dog among those classes. 
    * Test how the `dog face detector` detect `human faces` and `dog faces` 
* Build CNN to classify dog breeds 
    * Splt train, valid, test datasets  
        - define transforms for 3 datasets, respectively
        - load data
        - define dataloader
    * Build CNN architecture: convolutional layers + fully connected layers
    * train and valid model: forward pass/ backward prop/ print valid loss
    * test accuracy
* Build CNN to classify dog breeds using transder learning
    - load vgg16 pretrained model (features layers (conv) + classifier layers (fc))
    - Define data loader: resize images to dim=224x224 as vgg16 model accept that dim. 
    - Freeze the training for all features layers.  
    - Replace the last layer in classifier layers with a new linear layer. The new layer has the 133 dog breed classes as outputs. 
    - train and valid model: forward pass/ backward prop/ print valid loss
    - test accuracy
    - use trained model to predict dog breed
        - Load image files
        - transform: resize, ToTensor, unsqueeze
        - model predict dog breed
* Pipeline, use img_path as input 
    - combine above functions: human_face_detector / dog_face_detector / dog_breed_predictor to show image results  
    
    

## Approaches to improve accuracy
- There are possible methods to further improve the accuracy:
    - Increase the image datasets for training. Data augmentation such as image rotation, flipping, etc would help inprove model performance. 
    - Tune Hyperparameters: weigth initialization, number of epoches, learning rate, dropouts, batch size, more convolutional layers (filters) to detect features, filter size, stride, etc.
    - This model wrongly detected a cat as human. Train other species may also improve the accuracy by distinguishing dogs from other species. 
