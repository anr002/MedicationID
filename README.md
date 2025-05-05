# MedicationID, Medication Classification – Data Exploration

This project uses the C3PI RxImage dataset from the NIH to build a model that can classify pills based on their image.  
The end goal is to give the model a picture of a pill and have it predict the correct medication.

---

##  Dataset Overview

- **Source**: [[NIH C3PI RxImage Project](https://lhncbc.nlm.nih.gov/project/c3pi-computational-photography-project-pill-identification)](https://data.lhncbc.nlm.nih.gov/public/Pills/)
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

---

##  Future Preprocessing Steps 

Here’s how we prepare the data for training:

1. **Select 1 image per pill**  
   We only keep the highest resolution available so we dont overfit

2. **Crop label panel**  
   The bottom of each image has printed info. We can crop ~15–20% off the bottom to avoid cheating via text.

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
