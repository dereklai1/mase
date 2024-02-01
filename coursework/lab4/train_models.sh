

./ch train mnist_lab4_relu mnist \
        --accelerator gpu \
        --project-dir /home/derek/mase/coursework/lab4 \
        --project mnist_train \
        --max-epochs 20 \
        --batch-size 256 \
        --learning-rate 0.0001 \
        --weight-decay 0.00000001

# ./ch train mnist_lab4_leakyrelu mnist \
#         --accelerator gpu \
#         --project-dir /home/derek/mase/coursework/lab4 \
#         --project mnist_train \
#         --max-epochs 20 \
#         --batch-size 256 \
#         --learning-rate 0.0001 \
#         --weight-decay 0.00000001
