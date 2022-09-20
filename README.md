## Transition to Adulthood for Young People with Intellectual or Developmental Disabilities: Emotion Detection and Topic Modeling

This repository contains the code for corpus-based emotion detection and NMF topic modeling. 

## Emotion detection 

We used [NRC](https://saifmohammad.com/WebPages/AccessResource.htm) Word-Emotion Association Lexicon for emotion identification. You can apply NRC lexicon using the `get_emo_words` function and then calculate emotion distribution, average emotion intensity and return the most common emotion-associated words. More details are avaliable in our [paper](https://doi.org/10.1007/978-3-031-17114-7_21) 

## Topic modeling

The topic modeling is based on Non-Negative Matrix Factorization model from [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html#sklearn.decomposition.NMF). See an example of its work below:

<img width="768" alt="Picture1" src="https://user-images.githubusercontent.com/47871121/191172377-e4a86615-1d92-4cb3-b9f5-07d9807230b5.png">

## Citation

Please cite our paper as below:
@InProceedings{10.1007/978-3-031-17114-7_21,
author="Liu, Yan and Laricheva, Maria and Zhang, Chiyu and Boutet, Patrick and Chen, Guanyu and Tracey, Terence and Carenini, Giuseppe and Young, Richard",
title="Transition to Adulthood for Young People with Intellectual or Developmental Disabilities: Emotion Detection and Topic Modeling",
booktitle="Social, Cultural, and Behavioral Modeling",
year="2022",
publisher="Springer International Publishing",
isbn="978-3-031-17114-7"
}


