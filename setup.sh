cd SRC_swig
swig -python LKH.i
python3 setup.py build_ext --inplace
