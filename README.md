# MedicationID, Medication Classification – Data Exploration

This project uses the C3PI RxImage dataset from the NIH to build a model that can classify pills based on their image.  
The end goal is to give the model a picture of a pill and have it predict the correct medication.

---

##  Dataset Overview

- **Source**: [NIH C3PI RxImage Project](https://datadiscovery.nlm.nih.gov/Drugs-and-Chemicals/Computational-Photography-Project-for-Pill-Identif/5jdf-gdqh/about_data)
            - Direct download: https://data.lhncbc.nlm.nih.gov/public/Pills/rximage.zip
- **Data Structure**:
  - `table.csv`: Contains metadata for each pill (drug name, NDC, image paths by resolution)
  - `gallery/`: Image folder with 6 resolutions per pill (120, 300, 600, 800, 1024, original)

---

##  Data Exploration Summary

- Total labeled entries: ~4,332
- Unique drug names (classes): ~2,100
- All images are `.jpg`
- Each pill can have multiple image resolutions
- No missing images
- Images are not uniform in size due to different resolutions
- Plotted 40 sample pills with drug names
- Bottom of each image contains a blue metadata label that will have to be cropped out
![00002-4772-90_RXNAVIMAGE10_8C16C656](https://github.com/user-attachments/assets/28519e29-af26-4ef0-9156-66ef86df283b)

---

##  Future Preprocessing Steps 

Here’s how we prepare the data for training:

1. **Select 1 image per pill**  
   We only keep the highest resolution available so we dont overfit

2. **Crop label panel**  
   The bottom of each image has printed info. We can crop ~15–20% off the bottom to avoid cheating via text.
![00007-4139-20_NLMIMAGE10_9C18CE46](https://github.com/user-attachments/assets/5a4f296b-1ad5-4ac9-893e-49fad4f23574)

3. **Resize**  
   All images resized to 224x224 

4. **Normalize**  
   Normalize images

5. **Label encoding**  
   Drug names are encoded into integers (classification target).

6. **Train/Test Split**  
   Dataset split into 80% training / 20% test.


---

##  Environment Setup

```python
# !pip install opencv-python matplotlib pandas pillow

#See notebook for more
