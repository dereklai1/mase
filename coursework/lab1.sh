# Epochs
./ch train jsc-tiny jsc --max-epochs 5 --batch-size 128
./ch train jsc-tiny jsc --max-epochs 10 --batch-size 128
./ch train jsc-tiny jsc --max-epochs 20 --batch-size 128
./ch train jsc-tiny jsc --max-epochs 40 --batch-size 128

# Learning rate & batch size
./ch train jsc-tiny jsc --max-epochs 20 --batch-size 64 --learning-rate 0.00001
./ch train jsc-tiny jsc --max-epochs 20 --batch-size 256 --learning-rate 0.00001
./ch train jsc-tiny jsc --max-epochs 20 --batch-size 1024 --learning-rate 0.00001

./ch train jsc-tiny jsc --max-epochs 20 --batch-size 64 --learning-rate 0.001
./ch train jsc-tiny jsc --max-epochs 20 --batch-size 256 --learning-rate 0.001
./ch train jsc-tiny jsc --max-epochs 20 --batch-size 1024 --learning-rate 0.001

./ch train jsc-tiny jsc --max-epochs 20 --batch-size 64 --learning-rate 0.1
./ch train jsc-tiny jsc --max-epochs 20 --batch-size 256 --learning-rate 0.1
./ch train jsc-tiny jsc --max-epochs 20 --batch-size 1024 --learning-rate 0.1
