# randomised_spatial_smoothing_with_majority_voting
A suggested defence against adversarial attacks on image recognition employing majority voting of classifiers with randomized spatial smoothing.

Install Adversarial Robustness Toolbox from here : https://github.com/Trusted-AI/adversarial-robustness-toolbox/wiki/Get-Started#setup

Move the downloaded files from this git repo to the "adversarial-robustness-toolbox/notebooks/" directory.

The preparation.py file is to be run only once, to download the necessary image dataset from Imagenet along with the labels.

Run through the Randomised_Spatial_Smoothing_With_Majority_Voting.ipynb file to understand the example.

This defence applies spatial smoothing of randomised window sizes to the input image, thus generating multiple images, and the model predicts on each generated image, returning weighted average prediction scores of the class selected by majority voting.

The model was tested with 10,000 images downloaded from https://www.kaggle.com/datasets/priyerana/imagenet-10k, which in turn was randomly selected from the orignal challenge dataset containing about 1.43m images hosted by Imagenet at https://www.kaggle.com/competitions/imagenet-object-localization-challenge.
The test results can be found in the Test_Results folder, and the corresponding testing code can be found in the Test_Codes folder. Use the Test_Analysis.ipynb file to visualise the results.
