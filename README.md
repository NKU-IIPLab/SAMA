# SAMA
the source code for ACL2020 paper: Hiring Now: A Skill-Aware Multi-Attention Model for Job Posting Generation

the environment:

python==3.7.6

torch==1.2.0

cudatoolkit==10.0.130

1. the trained model and data: https://pan.baidu.com/s/1C1Ov7tKXPsp46anGQ0M9hQ  6ie4

   download the data and model.

2. ```bash
   unzip data_model.zip
   mv *pt SAMA/dataset
   mv pretrained_w2v SAMA/dataset
   mv trainedmodel SAMA/trained_model
   ```

3. train or just reproduce the results

   ```bash
   python train.py --file_dir SAMA/dataset     # for training
   python reproduce.py --file_dir SAMA/dataset    # just reproduce
   ```

