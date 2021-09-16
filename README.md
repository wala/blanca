### Can Machines Read Coding Manuals Yet? – A Benchmark for Building Better Language Models for Code Understanding

Code understanding is an increasingly important application of AI. A fundamental aspect of understanding code is understanding text about code, e.g., documentation  and forum discussions.  Pre-trained language models (e.g., BERT) are a popular approach for various NLP tasks, and there are now a variety of benchmarks, such as GLUE, to help improve the development of such models for natural language understanding. However, little is known about how well such models work on textual artifacts about code, and we are unaware of any systematic set of downstream tasks for such an evaluation.  In this paper, we derive a set of benchmarks (BLANCA - Benchmarks for LANguage models on Coding Artifacts) that assess code understanding based on tasks such as predicting the best answer to a question in a forum post, finding related forum posts, or predicting classes related in a hierarchy from class documentation.  We evaluate performance of current state-of-the-art language models on these tasks and show that there is significant improvement on each task from fine tuning.  We also show that multi-task training over BLANCA tasks help build better language models for code understanding.


### Task List:
- **Forum Answer Ranking (R)**:  Some answers on forums have many votes or are selected as the best relative to others. Can language models predict the best answers?
- **Forum Link Prediction (L)**: Users of forum posts often point to other similar posts, which reflect semantically related posts compared to random pairs.  Can language models predict links?
- **Forum to Class Prediction (F)**:  Key features of classes or functions often get discussed in forum posts.  Do language models discriminate related posts and class documentation from unrelated ones?
- **Class Hierarchy Distance Prediction (H)**: Code is often organized into class hierarchies.  Do embedding distances from language models reflect class distances in the hierarchy?
- **Class Usage Prediction (U)**: Similar code is often used in similar ways.  Are embedding distances smaller for documentation about classes that are used similarly, and larger for dissimilar ones?


### Models:

| Model name                            | Fine-tuning task           |
|---------------------------------------|----------------------------|
| Universal Sentence Encoder | N/A                        |
| BERT-NLI                  | Sentence similarity        |
| DistilBERT-paraphrasing   | Paraphrase detection       |
| xlm-r-paraphrase-v1       | Paraphrase detection       |
| mmsmarco-DistilRoBERTa    | Information Retrieval      |
| [BERTOverflow](https://github.com/lanwuwei/BERTOverflow)                | StackOverflow/NER          |
| [CodeBERT-mlm](https://huggingface.co/microsoft/codebert-base-mlm)              | NL-PL pairs in 6 languages |



### BLANCA Evaluation

### Dataset
The data for all five tasks are available here https://archive.org/details/blanca_202108 (with a Creative Commons Attribution-ShareAlike 4.0 International License, similar to StackExchange data dump)

To download the dataset, 
```buildoutcfg
cd data
bash ./download.sh
```

### Task Evaluation
Ahead of running any task, please set your PYTHONPATH as follows 
```buildoutcfg
export PYTHONPATH=`pwd`/tasks:$PYTHONPATH
```


- **Forum Answer Ranking (R)**

  To run on a standard non-finetuned model
   ```buildoutcfg
   cd tasks/ForumAnswerRanking/
   python -u rank_answers.py --eval_file ../../data/stackoverflow_data_ranking_v3_test.json --embed_type <embed_type> --model_dir <checkpoint_path>  
   ```
  where embed_type can be any of the following: USE, bert, roberta, distilbert, distilbert_para, xlm, or msmacro. In this case, you do not have to provide a value for `--model_dir`.
  
  To run using a checkpoint, e.g. BERTOverflow or CodeBERT, set embed_type to finetuned and provide the checkpoint path using `--model_dir` argument.


- **Forum Link Prediction (L)**: 
  Similar to Forum Answer Ranking, this task can be run using
  ```buildoutcfg
   cd tasks/ForumLinkPrediction/
   python test_linked_posts.py --eval_file ../../data/stackoverflow_data_linkedposts__testing.json --embed_type <embed_type> --model_dir <checkpoint_path>
   ```
- **Forum to Class Prediction (F)**:  
    ```buildoutcfg
   cd tasks/ForumToClassDocumentation/
  python test_class_posts.py --eval_file ../../data/class_posts_test_data_v3.json --embed_type <embed_type> --model_dir <checkpoint_path>
   ```
  
- **Class Hierarchy Distance Prediction (H)**: 
  ```buildoutcfg
   cd tasks/hierarchyAnalysis/
   python hierarchy_stats.py --eval_file ../../data/hierarchy_test.json --docstrings_file ../../data/merge-15-22.2.format.json --classmap ../../data/classes.map --classfail ../../data/classes.fail --embed_type <embed_type> --model_dir <checkpoint_path>
   ```
  
- **Class Usage Prediction (U)**: 
    ```buildoutcfg
   cd tasks/usageAnalysis/
   python staticAnalysis_regression.py --docstrings_file ../../data/merge-15-22.2.format.json --classmap ../../data/classes.map --eval_file ../../data/usage_test.json --embed_type <embed_type> --model_dir <checkpoint_path>
   ```


### Multi-Task Training

```buildoutcfg
python -u multi_task_train.py --model_name  --model_save_path <desired_checkpoint_folder> --data_dir ../../data/ --tasks <train_tasks> --validate <validate_tasks> --finetuned
```
where `<desired_checkpoint_folder>` is the folder where the training script will save the checkpoint, `<train_tasks>` is the task(s) used for training and `<validate_tasks>` is the task used for validation. 
`<train_tasks>` can be a combination of {hierarchy,linked_posts,class_posts,posts_rank,usage} where `<validate_tasks>` can be any of them. 

For example, to perform multi-task training on CodeBert using RFLH and test on ranking, one can use
```buildoutcfg
python -u multi_task_train.py --model_name microsoft/codebert-base-mlm --model_save_path checkpoint_microsoft_codebert_mlm_rflh_train_r_valid_ --data_dir ../../data/ --tasks posts_rank,class_posts,linked_posts,hierarchy, --validate posts_rank
```
To do the same using a checkpoint from a pretrained model; e.g. BertOverflow, change `model_name` to refer to the checkpoint folder instead. 


### Publications<a name="papers"></a>
* If you use Graph4Code in your research, please cite our work:

 ```
@misc{abdelaziz2021machines,
      title={Can Machines Read Coding Manuals Yet? -- A Benchmark for Building Better Language Models for Code Understanding}, 
      author={Ibrahim Abdelaziz and Julian Dolby and Jamie McCusker and Kavitha Srinivas},
      year={2021},
      eprint={2109.07452},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
 ```
 
### Questions
For any question, please contact us via email: ibrahim.abdelaziz1@ibm.com, kavitha.srinivas@ibm.com, dolby@us.ibm.com
