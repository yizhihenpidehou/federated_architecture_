# federated_architecture_
This repository is established on February 28, 2023, for the purpose of comparing model training under different network architectures

./data : Store data sets, such as nowplaying. It should be noticed that Nowplaying dataset includes nowplaying_test_sample.txt 
and nowplaying_valid_sample.txt, they are used to generate negative samples for validation or test, so the size of them should equal to
corresponding nowplaying.valid.inter and nowplaying.test.inter. 


./FMLP_Rec: Recommendation model. The usage can be found in the README.md of this directory.
    ./output: Store logs(.txt) and model(.pt) 
    ./datasets.py: Definition of FMLPRecDataset class and neg_sample function. If you do not have nowplaying_test_sample.txt and nowplaying_valid_sample.txt, this class will randomly generate a negative sample for 
each data.
    ./models.py: Model structure, using Pytorch
    ./modules.py: Model-related modules
    ./trainers.py: Definition of FMLPRecTrainer, it has two modes, train mode and validation mode, implementing in function iteration
    ./utils.py: Some useful functions for model to train, valid and test. 

main.py: Latest version of federated learning version baseline
trainers.py: Latest version of federated learning version trainer