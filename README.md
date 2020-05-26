# SAMA
 accepted by ACL2020: Hiring Now: A Skill-Aware Multi-Attention Model for Job Posting Generation

the environment:

+ python==3.7.6

+ torch==1.2.0

+ cudatoolkit==10.0.130

  

1. the trained model and data: [[url](https://drive.google.com/open?id=15bQqsOTVZrVbi-ivxfiYAALHohYQnt6P)]

**数据仅限于学术目的，不可用于商业活动。The data is only for academic purposes.**

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

## data description
+ src -- the text of job description  
+ srcid -- the word id of job description  
+ tgt -- the text of skill requirement  
+ tgtid -- the word id of skill requirement  
+ skillnet -- the recommed skill text of skill knowledge graph  
+ skillnetid -- the recommed skill word id of skill knowledge graph  
+ skilltgt -- the extracted skill text of skill requirement  
+ skilltgtid -- the extracted skill word id of skill requirement  

