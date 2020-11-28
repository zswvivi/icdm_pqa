# Requirements
- python3.6
- tensorflow-gpu 1.11
   - GPU: `pip install tensorflow-gpu==1.11.0`
- tensorflow_hub 0.1.1
- nltk,numpy, pandas, json, sklearn

(Note: our code is developed based on original Bert's implementation [original](https://github.com/google-research/bert))
[Annotated_Data](https://github.com/zswvivi/icdm_pqa/blob/master/data/ICDM_Annotated_Data.csv) <br>

# Data
1. The original dataset are from [dataset link](http://cseweb.ucsd.edu/~jmcauley/datasets.html), including Amazon QA and reviews data.
2. In our work, we have chosen 7 categories, namely Tools_and_Home_Improvement, Patio_Lawn_and_Garden, Automotive, Cell_Phones_and_Accessories, Health_and_Personal_Care, Sports_and_Outdoors, Home_and_Kitchen.
3. In order to replicate our results, you need to download QA (such as QA_Tools_and_Home_Improvement.json.gz etc) and review dataset (such as reviews_Tools_and_Home_Improvement.json.gz etc) in all of these categories.
4. Pleae keep downloaded files in folder "data".

# Dataset preprocessing
python data_preprocess.py  

# Training and Prediction
1. Cross-domain training FLTR: python FLTR_1st.py 
2. Fine-tuning FLTR for each category, for example:  python FLTR_2rd.py Home_and_Kitchen
3. Cross-domian training BERTQA: python BertQA.py ALL
4. Fine-tunning BERTQA for each category, for example: python BertQA.py Home_and_Kitchen

# Publication

Zhang, Shiwei; Lau, Jay Han; Zhang, Xiuzhen; Chan, Jeffrey; Paris, Cecile. Discovering Relevant Reviews for Answering Product-related Queries. In: IEEE 19th International Conference on Data Mining; November 2019;  Beijing, China. ICDM; 2019.
