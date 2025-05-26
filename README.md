# MedicationID

## Milestone 3

### Summary
For milestone 3, I worked on preprocessing the NIH C3PI RxImage dataset to classify pills based on their images. I focused on getting the data cleaned, setting up the first model, and preparing for evaluation.

### 1: Major Preprocessing
- Loaded `table.csv` with pandas since I was getting timeout errors with Spark.
- Iterated over each `RXNAV ORIGINAL` image.
- Resized images to 224x224.
- Normalized pixel values (divided by 255).
- Flattened each image into a feature vector.
- Saved the processed data into a parquet file to feed into Spark.

### 2: First Model
- Loaded parquet data into Spark.
- Used `StringIndexer` to encode drug names.
- Used `VectorAssembler` to combine features.
- Trained a `RandomForestClassifier` with 20 trees.

### 3: Evaluation
- Split the data 80/20 for training and testing.
- Evaluated model accuracy using `MulticlassClassificationEvaluator`.
- Then I printed test accuracy and showed some example predictions.

### 4: Where Does the Model Fit?
I planned on using Random Forest as a quick first try but because the preprocessing had to be restarted multiple times hours in, I wasn’t able to get the model trained on time. 
That said, I know Random Forest likely wouldn’t perform that well given the limited set of flattened pixel features. 
If time permitted, I believe using a multilayer perceptron or CNN would give better performance because they can work better with image data.


### 5: README and Files
- `Milestone3.ipynb` - Notebook with preprocessing + Spark code.
- `README.md`

### 6: Conclusion
I set up all the preprocessing and modeling pieces, but because of how long the image processing took, I couldn’t finish running the Random Forest by the deadline. 
However, the plan is laid out and ready to work forward. Next steps would be improving the input features, tuning models, and eventually moving to another model that can better handle the image data.

![image](https://github.com/user-attachments/assets/82c2f82e-625d-4ed2-8af3-ce6b9fb49869)
