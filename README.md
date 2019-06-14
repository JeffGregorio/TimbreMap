# TimbreMap
A system for learning timbre-based mappings from low-dimensional controllers to equal or higher-dimensional synthesizer parameter spaces. Includes trained models for a set of example Max/MSP patches. 

![System Overview](/images/overview_and_runtime.png)

**Left**: Overview of system during training and runtime. Encoded training examples in X are used to predict values in parameter space P. Training yields a normally-distributed, 3-dimensional latent encoding space Z with predictive dimensions. 

Post-training, data **X** is encoded into the latent space Z, which is re-oriented using principal component analysis (PCA) into Z\*, whose mean and standard deviation, along with PCA basis vectors and regressor weights are exported for the runtime model.

**Right**: At runtime, a vector of coordinates in a uniform control space C are scaled into the normally-distributed Z\* using the normal CDF, the PCA projection is inverted to place the vector in Z, and from Z a regressor consisting of a single dense layer infers values in parameter space P.

Each layer of the runtime model is invertible, so updates made in parameter space P are projected backward into the latent space Z, oriented into Z\*, and scaled to the control space C.

## Dependencies 

### Example patch, data generation, and runtime model dependencies
* [Max 8](https://cycling74.com/)
* [BLOCKS package](https://cycling74.com/feature/roliblocks)
* [jg.MaxSynthLib](https://github.com/JeffGregorio/jg.MaxSynthLib)
* [jg.MaxExternals](https://github.com/JeffGregorio/jg.MaxExternals)

### Training dependencies
* Python dependencies are listed in `requirements.txt`.

Note: this system has only been tested under Mac OS 10.14, using Python 3.6.8, Max 8, Max SDK 8.0.3, and BLOCKS 1.2.7.

## Max/MSP Patches
Example patches are provided in `/patches`. 

### Generating data
Generation of training data is handled by the javascript code provided in `jg.MaxSynthLib/timbremap/generator_rand.js` (see dependencies list), which generates parameters and controls note on/off, and generates messages to control a `sfrecord~` instance. It takes as arguments the number of parameters, number of examples, data directory, sustain time, and release time. For example, 

`js generator_rand.js 4 10000 data/test 200 100`

generates random values [0-127] for 4 parameters; 10,000 examples saved to data/test (relative to the patch's directory), where each note is sutained for 200ms, and WAV files are terminated 100ms after release.

See any single patch or `gen` version for an example configuration of these blocks.

#### Notes: 
* A data directory specified to `generator_rand.js` must exist before generating data, and it must contain a `wavs` directory for storing the wav files (e.g. in the above example, the patch's directory must be initialized with an empty `data/test/wavs` directory or it will fail).
* Once started all parameter values are generated at once, and a `labels.csv` is saved to the data directory, allowing data generation to be stopped and resumed at any time after (e.g. `resume 3360` will generate examples, starting at row 3360 of an existing `labels.csv` file)

### Running trained models
See example patches. Models are loaded by passing `load model_dir/timbremap` to the external `jg.timbremap`. 


## Computing Features
Prior to training models, use the `compute_melspecs.py` script to compute features from a directory of generated WAV files.

Usage:  
`python compute_melspecs.py [data_dir]`

Computes an N x F x T dataset of Mel-scaled spectrograms with F frequency bins and T time steps, exported to `data_dir/features.npy`. Examples should be in WAV format, contained in `data_dir/wavs`. 

#### Note:
* Occasionally, `sfrecord~` will write a corrupted WAV file and `compute_melspecs.py` will fail with `ValueError: There aren't any elements to reflect in axis 0 of 'array'`, in which case you can re-generate the example by sending `resume #` to `generate_rand.js`, (where `#` is the corrupted example), followed by `stop` after the example has been re-generated.

## Training Models

Train default configurations of generative/discriminative models with encoders based on Deep Neural Networks (DNN), Convolutional Neural Networks (CNN), Long Short Term Memory Networks (LSTM). Model architectures and parameters are defined in `util/models.py`.

`usage: train.py [-h] (--dnn | --cnn | --lstm) [--gen] [--pca]`
                `[--epochs EPOCHS] [--batch BATCH] [--latent_size LATENT_SIZE]`
                `data_dir model_dir`

*Example: train a model using a generative LSTM encoder, and re-orient the 3D latent space using PCA. Train for a default 10 epochs with batches of 32 examples*

`python train.py --lstm --gen --pca patches/subtractive/lfo4/data/fm patches/subtractive/lfo4/models/fm`

Exports runtime model parameters to `model_dir/timbremap`, as well as keras models in json and h5 formats, and training data projected into the original and PCA-reoriented latent space.

Data directories should contain `features.npy` and `labels.npy`. Running any of the training scripts on a data directory for the first time will generate a 90% training, 10% testing partition `partition.npy`, which will be reused unless it is deleted.

If a data directory does not contain `features.npy` and `labels.npy`, the training scripts will recursively search sub-directories for features and labels, and train on a single dataset consisting of all `features.npy` and `labels.npy` matrices concatenated row-wise. For example, if the directory `patches/subtractive/lfo4/data` contains data sub-directories for FM, PWM, and FCM, we can train a universal model on data from all modulation types with  

`python train.py --lstm --gen --pca patches/subtractive/lfo4/data patches/subtractive/lfo4/models/universal`  
















