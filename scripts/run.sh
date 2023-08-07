cd model

#* fine-tune the lamb hyperparameter
# python traintest_MemGCRN.py --lamb 0.01 --temp 0.1   
python traintest_MemGCRN.py --lamb 0.1 --temp 1.0
python traintest_MemGCRN.py --lamb 1 --temp 1.0
python traintest_MemGCRN.py --lamb 10 --temp 1.0 # worse

# python traintest_MemGCRN.py --lamb 0.01 --temp 0.1 --contra_denominator
python traintest_MemGCRN.py --lamb 0.1 --temp 0.1 --contra_denominator
python traintest_MemGCRN.py --lamb 1 --temp 0.1 --contra_denominator
python traintest_MemGCRN.py --lamb 10 --temp 0.1 --contra_denominator

#* fine-tune the temperature hyperparameter  
# python traintest_MemGCRN.py --temp 0.5 --lamb 0.01
# python traintest_MemGCRN.py --temp 1 --lamb 0.01

python traintest_MemGCRN.py --temp 0.5 --lamb 0.01 --contra_denominator
python traintest_MemGCRN.py --temp 1 --lamb 0.01 --contra_denominator