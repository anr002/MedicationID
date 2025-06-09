# MedicationID – Pill Classification from Images

This project tries to classify pills using images from the NIH C3PI RxImage dataset. The idea is to be able to take a picture of a pill and have the model tell you which medication it is.

---

## Introduction

The RxImage dataset has photos of thousands of medications, but this problem turned out to be harder than I expected. Many pills look really similar, and the images vary a lot — different lighting, different backgrounds, different sizes. Some pills only have 1 or 2 images. And trying to get a model to learn from that is tough.

I used Spark to handle the processing and training since the dataset is so big, but I ran into a lot of problems with memory, timeouts, and crashing sessions. I wasted hours of time running something that silently failed or gave me nothing useful. A lot of this is explained more in the report.

---

## What’s in Here

- `FinalSubmission.ipynb`: Switched over to C3PI images and trained a logistic regression model on the top 20 drug classes.
- `WrittenReport.pdf`: Written report

---

## What I Did

### Preprocessing

- Used `directory_consumer_grade_images.txt` to get the image paths
- Downloaded images into folders based on drug name
- Resized images to 224x224
- Normalized them to 0–1
- Flattened them into vectors
- Encoded the drug names as numbers
- Saved everything to a parquet file

### Early Tries and Problems

At first, I used the original RxImage dataset which had 6 resolutions per pill. That had very few classes with enough examples to train anything. Some drugs had only 2 images. That wasn’t going to work.

So I switched to the consumer-grade C3PI images. But that brought new issues as the image quality was a mess. Some had calibration cards, others didn’t. Some were blurry, others were washed out or had shadows. Trying to preprocess all of them the same way was impossible. Nothing was consistent.

Also, a lot of time was wasted dealing with Spark sessions crashing or dying. Sometimes I’d preprocess images for 10 hours and then realize something broke halfway. A lot of this was days of trial and error.

---

## Final Model

I ended up using logistic regression on the top 20 most common drug classes. I trained on 80% of the data and tested on 20%. The features were just flattened image vectors — no fancy deep learning.

Results for original high quality data but limited classes:

Random Forest:
training random forest
accuracy: 0.3125
f1 score: 0.2979
weighted precision: 0.3255
weighted recall: 0.3125

Logistic Regression Results:
accuracy: 0.2812
f1 score: 0.3237
weighted precision: 0.4453
weighted recall: 0.2812
![image](https://github.com/user-attachments/assets/c011fa86-d71f-4690-81ae-1ef4c28b4b61)




Logisitic Regression Using Consumer Images:
accuracy: 0.2625
f1 score: 0.1933
weighted precision: 0.2003
weighted recall: 0.2625

![image](https://github.com/user-attachments/assets/37ceb814-9de8-448a-bbf3-6a30e9ecba2d)


The model isn't great, but it runs, and it shows how this could work with better image features. If I had more time, I’d try a CNN instead. Flattened pixels just don’t cut it for this type of data.

Trying a CNN I got horrible score. Like I said the data was very bad.
Test Accuracy: 0.0264
Weighted F1 Score: 0.0057



                                        accuracy                           0.03      2234
                                       macro avg       0.00      0.01      0.00      2234
                                    weighted avg       0.00      0.03      0.01      2234

---

## Final Thoughts

This took way more time than expected because of how bad the raw data was and how fragile Spark is when it comes to memory and long-running jobs. Still, I got something working. I explained all of it in the report, including all the things I had to redo and what I’d improve if I kept going.
