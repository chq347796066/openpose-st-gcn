import torch

def a():
    frame_num=10
    filter_len=5
    filter=torch.zeros(frame_num, frame_num,1)
    for i in range(frame_num):
        for j in range(filter_len):
            if i+j-filter_len+1<0:
                continue
            else:
                filter[i][i+j-filter_len+1][0]=1
    print(filter)

a()
