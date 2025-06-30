python unlearn.py --exp baseline --inversion goae --inversion_image_path ./data/CelebAHQ/512 --target average --local --seed 0

python unlearn.py --exp guide --inversion goae --inversion_image_path ./data/CelebAHQ/512 --target extra --target_d 30.0 --local --adj --glob --seed 0

python evaluate_id.py --exp guide
python evaluate_fid.py --exp guide --seed 0