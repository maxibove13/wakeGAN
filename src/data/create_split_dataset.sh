# # go to data dir
cd data

# remove previous dataset
rm -r preprocessed/ux/*
rm -r preprocessed/uy/*
rm -r preprocessed/tracked

cd ..
# create dataset
./src/data/make_dataset.py
# split dataset into train and test
./src/data/split_data.py --ratio 0.9 0.1

cd data
# move train/test set to tracked dir
mv preprocessed/val preprocessed/test
mkdir preprocessed/tracked &&
mv preprocessed/train preprocessed/tracked
mv preprocessed/test preprocessed/tracked