Image augmentation GPU
======================


Wrapper for imgaug library to make some of its augmentations
 to work on GPU.
 
Currently does that using a pytorch based implementation.
Useful to speed up your trainings if you already use pytorch.



Requirements
-------

python 3.5+

pytorch 0.3.1 gpu

cudnn 8.0

imguag (some version)



Running the example
------

We provide a simple example showing the current implemented functionalities.
To run the example, run:

    python run_example.py
    
The example should plot the poor rat from imgaug over different
augmentation as following:



Note, there is a difference between gpu and cpu results,
since this wrapper does not use exactly the same parent system
for getting random seeds. The system was designed to
look more like pytorch.transforms interface. (PR to)


Running the Speed Test
-----------



    
Brief Explanation
-------------



Supported operations on GPU
--------------------
The results should be similar as imgaug except for #1

##### Image Augmentation
Add
Multiply
Dropout
CoarseDropout
ContrastNormalization
GaussianBlur
Grayscale


##### Image selection

Sometimes


The natural thing would be to adapt this to eliminate 
imgaug dependence and to make a pull request to pytorch.








Library benchmark for TITAN XP in a Core i7 multiprocessed
-----




### Want to apply augmentations to solve a cool problem ?
Check CARLA simulator
