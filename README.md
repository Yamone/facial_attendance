 ***Model training for mobilefacenet***
 1. Create a virtual environment.
 2. Installation make from requirement.txt.
 > pip install -r requirement.txt
 3. Do preprocess input raw image.(if required)
 > python preprocess.py --input_dir 'go_to_raw_image_directory' --output_dir final_processing_data    --crop_dim 'crop_image_size'#112
 4. Train data.
 > python train.py --data_dir "preprocess data directory"

***Run Server Command***
1. Change directory to the file path that exist app.py.
2. list virtual env
> lsvirtualenv
3. activate the virutalenv
> workon facialAttendance-env
4. activate bash_profile
> source ~/.bash_profile
5. Run python file
> python app.py
