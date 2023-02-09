# A Deep Multi-Embedding Model for Mobile Application Recommendation
This is a release of the proposed model that combines the BERT-based embedding model and deep matrix factorization method for rating prediction.

In total, our research developed 13 program files.
Description of each program file:
1. Datapreprocessing.ipynb
   Preliminary processing of the collected data includes screening all data with at least six or more comments from the same user and pre-processing data for the review content.
2. split_dataset.ipynb & split_origin_review.ipynb
   Split the dataset into training/testing/validation data
3. review2bert.py
   Convert review to BERT embedding vector.
4. review2roberta.ipynb
   Convert review to RoBerta embedding vector.
5. split_dataset_add_emb.ipynb
   According to the index file of the disassembled data set above, obtain the corresponding embedding for training/validation/test data.
6. convert_tensor.ipynb
   Split the three files (training/validation/test) produced by the previous step into different fields, and convert them to fit the tensor-flow format for subsequent use.
7. reindex-appId.ipynb
   The AppID in the stored file was not reordered after cleaning, so this file is executed and recovered.
8. rating_learning_model.ipynb
   The rating learning model.
9. review_learning_1rv_BERT_model.ipynb
   The RV1-BERT model
10. review_learning_1rv_RoBERTa_model.ipynb
   The RV1-RoBERTa model
11. review_learning_2rv_RB_model.ipynb
   The RV2 model
12. RARV2_combined.ipynb
   The proposed full model RARV2
