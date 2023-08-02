cd model

#* fine-tune the lamb hyperparameter
# python traintest_MemGCRN.py --lamb 0.01 --temp 0.1  worse than baseline
python traintest_MemGCRN.py --lamb 0.1 --temp 0.1
python traintest_MemGCRN.py --lamb 1 --temp 0.1
python traintest_MemGCRN.py --lamb 10 --temp 0.1

#* fine-tune the temperature hyperparameter
# python traintest_MemGCRN.py --temp 0.1 --lamb 0.01  worse than baseline
python traintest_MemGCRN.py --lamb 0.5 --lamb 0.01
python traintest_MemGCRN.py --lamb 1 --lamb 0.01