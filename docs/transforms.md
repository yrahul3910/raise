# `raise.transforms`

This module provides a set of pre-processing transforms for data. An instance of `Transform` is created, passing in the name of the transform (with an option for random parameter settings), and then the `apply` method is called, passing in a `Data` instance. The following transforms are supported:
   
* `normalize`  
* `standardize`  
* `minmax`  
* `maxabs`  
* `robust`  
* `kernel`: calls `KernelCenterer` from `sklearn`  
* `smote`  
* `cfs`: implements Correlation-based Feature Selection  
* `wfo`: implements weighted fuzzy oversampling  
* `rwfo`: implemented radially weighted fuzzy oversampling  
* `tf`: term frequency  
* `tfidf`: tf-idf  
* `hashing`: hashing vectorizer  
* `lda`: latent Dirichlet allocation  
  
Additional transforms can be user-created by subclassing `Transform` and overriding the `apply` method.
