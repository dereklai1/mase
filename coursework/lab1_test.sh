# Epochs

CHECKPOINTS=../mase_output/jsc-tiny_classification_jsc_2024-01-15/software/training_ckpts

./ch test jsc-tiny jsc --load $CHECKPOINTS/best.ckpt --load-type pl

for i in {1..12}; do
./ch test jsc-tiny jsc --load $CHECKPOINTS/best-v$i.ckpt --load-type pl
done
