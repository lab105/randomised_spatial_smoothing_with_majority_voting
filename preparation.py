# Install ImageNet stubs:
!{sys.executable} -m pip install git+https://github.com/nottombrown/imagenet_stubs
!{sys.executable} -m pip install kaggle

#To use the Kaggle API, sign up for a Kaggle account at https://www.kaggle.com. Then go to the 'Account' tab of your user profile (https://www.kaggle.com/<username>/account) and select 'Create API Token'. This will trigger the download of kaggle.json, a file containing your API credentials. Place this file in the location ~/.kaggle/kaggle.json (on Windows in the location C:\\Users\\<Windows-username>\\.kaggle\\kaggle.json)
#Dataset of 10,000 Images from Imagenet
!kaggle datasets download -d priyerana/imagenet-10k

#Either use the following command to unzip followed by remove the compressed file, or you can do it manually. Specify the paths accordingly in the next steps.
!unzip imagenet-10k.zip && rm imagenet-10k.zip

#Getting the labels file
!kaggle competitions download imagenet-object-localization-challenge -f LOC_synset_mapping.txt
!readlink -f imagenet_subtrain/* > imagepaths.txt

#Moving all images to a single folder for easier access.
!mv imagenet_subtrain/*/* imagenet_subtrain/ && rmdir imagenet_subtrain/*