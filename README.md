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

Class Breakdown:

                                                  precision    recall  f1-score   support

                               ACARBOSE_TAB_50MG       0.00      0.00      0.00         8
                                ACARBOSE_Tablets       0.00      0.00      0.00         1
                                        ACCUPRIL       0.00      0.00      0.00         6
                           ACCUPRIL_10MG_TABLETS       0.00      0.00      0.00         4
ACETAMINOPHEN_and_CODEINE_PHOSPHATE_Tablets,_USP       0.00      0.00      0.00         9
                                       ADALAT_CC       0.00      0.00      0.00         5
                              ADVICOR_TAB1000-40       0.00      0.00      0.00         6
                            ADVICOR_TAB_500-20MG       0.00      0.00      0.00         4
                            ADVICOR_TAB_750-20MG       0.00      0.00      0.00         7
                             ALDACTAZIDE_Tablets       0.00      0.00      0.00         6
                               ALLEGRA-D_12_HOUR       0.00      0.00      0.00        12
                               ALLEGRA-D_24_HOUR       0.00      0.00      0.00         5
                            ALLEGRA_ODT_TAB_30MG       0.00      0.00      0.00         4
                              AMARYL_4MG_TABLETS       0.00      0.00      0.00        26
                                    AMBIEN_10_MG       0.00      0.00      0.00         5
                            AMBIEN_CR_TAB_12.5MG       0.00      0.00      0.00         6
                           AMBIEN_CR_TAB_6.25MGS       0.00      0.00      0.00         4
                 AMLODIPINE_BESYLATE_Tablets_USP       0.00      0.00      0.00         6
                                    APAP_CODEINE       0.00      0.00      0.00         4
                               AROMASIN_25MG_TAB       0.00      0.00      0.00         5
                          ARTHROTEC_50MG_TABLETS       0.00      0.00      0.00         6
                          ARTHROTEC_75MG_TABLETS       0.00      0.00      0.00         5
                      AVALIDE_150_12.5MG_TABLETS       0.00      0.00      0.00         4
                     AVALIDE_300_12.5_MG_TABLETS       0.00      0.00      0.00         9
                                AVANDAMET_2_1000       0.00      0.00      0.00         7
                              AVANDAMET_4_1000MG       0.00      0.00      0.00         7
                                       AVANDARYL       0.00      0.00      0.00        26
                                         AVANDIA       0.00      0.00      0.00        17
                                AVANDIA_TAB_8_MG       0.00      0.00      0.00         5
                            AVAPRO_150MG_TABLETS       0.00      0.00      0.00         9
                                   AVAPRO_300_MG       0.00      0.00      0.00         5
                                          AVELOX       0.00      0.00      0.00         5
                                AVELOX_400MG_TAB       0.00      0.00      0.00         4
                          AXERT_12.5MG_12&apos;S       0.00      0.00      0.00         6
                       AZATHIOPRINE_50MG_TABLETS       0.00      0.00      0.00         5
               BALSALAZIDE_DISODIUM_Capsules_USP       0.00      0.00      0.00         4
                                   BIAXIN_500_MG       0.00      0.00      0.00         9
                           BUPRENORPHINE_SUB_8MG       0.00      0.00      0.00        12
                              BUSPIRONE_10MG_TAB       0.00      0.00      0.00         8
             BusPIRone_HYDROCHLORIDE_Tablets_USP       0.00      0.00      0.00         5
                          CADUET_10_20MG_TABLETS       0.00      0.00      0.00         5
                           CADUET_5_10MG_TABLETS       0.00      0.00      0.00         6
                           CADUET_5_20MG_TABLETS       0.00      0.00      0.00         6
                          CALAN_SR_240MG_TABLETS       0.00      0.00      0.00         3
                                      CALCITRIOL       0.00      0.00      0.00         3
                                 CALCIUM_ACETATE       0.00      0.00      0.00         4
                       CARBAMAZEPINE_TABLETS_USP       0.00      0.00      0.00         8
                               CARDIZEM_LA_120MG       0.00      0.00      0.00         6
                          CARDIZEM_LA_180MG_TABS       0.00      0.00      0.00         4
                          CARDIZEM_LA_240MG_TABS       0.00      0.00      0.00         6
                          CARDIZEM_LA_360MG_TABS       0.00      0.00      0.00         4
                          CARVEDILOL_Tablets_USP       0.00      0.00      0.00        17
                     CELEBREX_CELECOXIB_Capsules       0.00      0.00      0.00         8
                         CELLCEPT_250MG_CAPSULES       0.00      0.00      0.00         6
                          CELLCEPT_500MG_TABLETS       0.00      0.00      0.00         4
               CEVIMELINE_HYDROCHLORIDE_Capsules       0.00      0.00      0.00         2
                                 CHANTIX_1MG_TAB       0.00      0.00      0.00         5
                               CHANTIX_TAB_0.5MG       0.00      0.00      0.00         7
                             CIALIS_10MG_TABLETS       0.00      0.00      0.00         5
                             CIALIS_20MG_TABLETS       0.00      0.00      0.00         3
                                  CIALIS_5MG_TAB       0.00      0.00      0.00        11
                          CILOSTAZOL_Tablets_USP       0.00      0.00      0.00         4
                                           CIPRO       0.00      0.00      0.00         6
                          CLARINEX-D_TAB_2.5-120       0.00      0.00      0.00        15
                          CLARINEX-D_TAB_5-240MG       0.00      0.00      0.00        15
                            CLARINEX_5MG_TABLETS       0.00      0.00      0.00         7
                          CLARITHROMYC_500MG_TAB       0.00      0.00      0.00         3
                               CLEOCIN_75MG_CAPS       0.00      0.00      0.00         4
                                 COARTEM_Tablets       0.00      0.00      0.00         3
                                    COMTAN_200MG       0.00      0.00      0.00         4
                                           COREG       0.00      0.00      0.00        33
                                        COREG_CR       0.00      0.00      0.00        13
                               COREG_CR_CAP_40MG       0.00      0.00      0.00         4
                               COREG_CR_CAP_80MG       0.00      0.00      0.00         3
                             CORTEF_10MG_TABLETS       0.00      0.00      0.00         5
                          COUMADIN_1MG_PINK_TABS       0.00      0.00      0.00         8
                       COUMADIN_2.5MG_GREEN_TABS       0.00      0.00      0.00         6
                      COUMADIN_2MG_LAVENDER_TABS       0.00      0.00      0.00         2
                           COUMADIN_3MG_TAN_TABS       0.00      0.00      0.00         5
                          COUMADIN_4MG_BLUE_TABS       0.00      0.00      0.00        11
                         COUMADIN_5MG_PEACH_TABS       0.00      0.00      0.00        13
                      COUMADIN_7.5MG_YELLOW_TABS       0.00      0.00      0.00        11
                                    COZAAR_100MG       0.00      0.00      0.00         6
                                    COZAAR_50_MG       0.00      0.00      0.00         5
   CREON_(PANCRELIPASE)_Delayed-Release_Capsules       0.00      0.00      0.00         3
                              CREON_24000UNT_CAP       0.00      0.00      0.00         6
                                       CREON_CAP       0.00      0.00      0.00         3
                  CREON_Delayed-Release_Capsules       0.00      0.00      0.00         1
                   CYCLOPHOSPHAMIDE_50MG_TABLETS       0.00      0.00      0.00         6
                               CYMBALTA_20MG_CAP       0.00      0.00      0.00         7
                               CYMBALTA_30MG_CAP       0.00      0.00      0.00         3
                               CYMBALTA_60MG_CAP       0.00      0.00      0.00        10
                                   Carbamazepine       0.00      0.00      0.00         6
                                      Carvedilol       0.00      0.00      0.00        14
                                  Clarithromycin       0.00      0.00      0.00         6
                 ClomiPHENE_CITRATE_Tablets,_USP       0.00      0.00      0.00         7
                                    DAYPRO_600MG       0.00      0.00      0.00         6
                       DEPAKOTE_250MG_PEACH_TABS       0.00      0.00      0.00         3
                           DEPAKOTE_500MG_TABLET       0.00      0.00      0.00         7
                       DEPAKOTE_SPRINKLES_125_MG       0.08      0.21      0.12        24
                                     DEPAKOTE_XR       0.00      0.00      0.00         2
                         DEPAKOTE_XR_250MG_WHITE       0.00      0.00      0.00         4
                                      DETROL_2MG       0.00      0.00      0.00         4
                               DETROL_LA_2MG_CAP       0.00      0.00      0.00         9
                               DETROL_LA_4MG_CAP       0.00      0.00      0.00         4
                        DEXAMETHASONE_0.5MG_TABS       0.00      0.00      0.00         6
                           DEXAMETHASONE_1MG_TAB       0.00      0.00      0.00         5
                              DEXAMETHASONE_2_MG       0.00      0.00      0.00        10
                           DEXAMETHASONE_4MG_TAB       0.00      0.00      0.00        10
                             DILANTIN_100MG_CAPS       0.00      0.00      0.00        11
                               DILANTIN_30MG_CAP       0.00      0.00      0.00         3
                           DILATRATE_SR_40MG_CAP       0.00      0.00      0.00         5
                                          DIOVAN       0.00      0.00      0.00         1
                            DIOVAN_160MG_TABLETS       0.00      0.00      0.00         7
                            DIOVAN_320MG_TABLETS       0.00      0.00      0.00         9
                             DIOVAN_80MG_TABLETS       0.00      0.00      0.00         6
                  DIOVAN_HCTZ_160_12.5MG_TABLETS       0.00      0.00      0.00         7
                      DIOVAN_HCTZ_160_25_TABLETS       0.00      0.00      0.00         4
                         DIOVAN_HCT_320_25MG_TAB       0.00      0.00      0.00         8
                      DIOVAN_HCT_80_12.5_TABLETS       0.00      0.00      0.00         4
                         DIOVAN_HCT_TAB_320_12.5       0.00      0.00      0.00         6
                                         DYAZIDE       0.00      0.00      0.00         7
                                    DYAZIDE_CAPS       0.00      0.00      0.00         9
                                   Dexamethasone       0.00      0.00      0.00        13
                     EFFEXOR_XR_150MG_BROWN_CAPS       0.00      0.00      0.00         6
                     EFFEXOR_XR_37.5MG_GREY_PINK       0.00      0.00      0.00         2
                             EFFEXOR_XR_75MG_CAP       0.00      0.00      0.00         2
                                 EFFIENT_Tablets       0.00      0.00      0.00         3
                                 ELIQUIS_Tablets       0.00      0.00      0.00         4
                      EMEND_(APREPITANT)_Capsule       0.00      0.00      0.00         1
                     EMEND_(APREPITANT)_Capsules       0.00      0.00      0.00         5
                   ENALAPRIL_MALEATE_Tablets_USP       0.03      0.18      0.06        22
                            ERY-TAB_250MG_EC_TAB       0.00      0.00      0.00         5
                                   ERY-TAB_333MG       0.00      0.00      0.00         5
                         ERYTHROMYCIN_BASE_250MG       0.00      0.00      0.00         2
                         ERYTHROMYCIN_BASE_500MG       0.00      0.00      0.00         5
                          ERYTHROM_ETH_TAB_400MG       0.00      0.00      0.00         4
                               ESTAZOLAM_Tablets       0.00      0.00      0.00         8
                                          EVISTA       0.03      0.50      0.05         4
                                 EXELON_3MG_CAPS       0.00      0.00      0.00         5
                                         EXFORGE       0.00      0.00      0.00         9
                                     EXFORGE_HCT       0.00      0.00      0.00         2
                                     EXFORGE_XCT       0.00      0.00      0.00         4
              Epitol_(CARBAMAZEPINE_Tablets_USP)       0.00      0.00      0.00        10
                             FAMCICLOVIR_Tablets       0.00      0.00      0.00         2
                                          FANAPT       0.00      0.00      0.00         5
                      FANAPT_ILOPERIDONE_Tablets       0.00      0.00      0.00         5
                            FLECAINIDE_100MG_TAB       0.00      0.00      0.00         3
                             FLECAINIDE_50MG_TAB       0.00      0.00      0.00         7
                                      Furosemide       0.00      0.00      0.00        10
                             GEODON_20MG_CAPSULE       0.00      0.00      0.00         7
                                GEODON_40MG_BLUE       0.00      0.00      0.00         7
                              GEODON_60MG_WHHITE       0.00      0.00      0.00         4
                          GEODON_ZIPRASIDONE_HCl       0.00      0.00      0.00         2
                               GLUCOPHAGE_500_MG       0.02      0.55      0.04        11
                              GLUCOPHAGE_Tablets       0.00      0.00      0.00         2
                             GLUCOPHAGE_XR_500MG       0.00      0.00      0.00         4
                                 GLYSET_25MG_TAB       0.00      0.00      0.00         5
                         GRISFULVIN_V_500MG_TABS       0.00      0.00      0.00         6
                               HYCAMTIN_CAPSULES       0.00      0.00      0.00         6
                                     HYOPHEN_TAB       0.00      0.00      0.00         3
                             HYZAAR_TAB_100-12.5       0.00      0.00      0.00        11
                               HYZAAR_TAB_100-25       0.00      0.00      0.00        12
                              HYZAAR_TAB_50-12.5       0.02      0.20      0.04        15
 IRBESARTAN_and_HYDROCHLOROTHIAZIDE_Tablets,_USP       0.00      0.00      0.00        10
                            ISENTRESS_400_MG_TAB       0.00      0.00      0.00         6
                          Isosorbide_Mononitrate       0.00      0.00      0.00         2
                             JANUMET_TAB_50-1000       0.00      0.00      0.00         7
                            JANUMET_TAB_50-500MG       0.00      0.00      0.00         8
             JANUMET_XR_Extended-Release_Tablets       0.00      0.00      0.00         4
                               JANUVIA_TAB_100MG       0.00      0.00      0.00        11
                                JANUVIA_TAB_25MG       0.00      0.00      0.00         5
                                JANUVIA_TAB_50MG       0.00      0.00      0.00         6
                              Janumet_XR_Tablets       0.00      0.00      0.00         2
                         KOMBIGLYZE_2.5-1000_TAB       0.00      0.00      0.00         9
                               K_TABS_10MEQ_TABS       0.00      0.00      0.00         5
                             LAMOTRIGINE_Tablets       0.00      0.00      0.00         4
     LAMOTRIGINE_Tablets_(chewable,_dispersible)       0.00      0.00      0.00         7
                              LASIX_20MG_TABLETS       0.00      0.00      0.00         6
                              LASIX_40MG_TABLETS       0.00      0.00      0.00         5
                         LEFLUNOMIDE_Tablets_USP       0.00      0.00      0.00        16
                                    LESCOL_20_MG       0.00      0.00      0.00         8
                          LESCOL_XL_80MG_TABLETS       0.00      0.00      0.00         5
                          LETROZOLE_Tablets,_USP       0.00      0.00      0.00         2
                  LEUCOVORIN_CALCIUM_Tablets_USP       0.00      0.00      0.00         4
                            LIPITOR_10MG_TABLETS       0.00      0.00      0.00        11
                             LIPITOR_20MG_TABLET       0.00      0.00      0.00         3
                            LIPITOR_40MG_TABLETS       0.00      0.00      0.00         5
                            LIPITOR_80MG_TABLETS       0.00      0.00      0.00         9
                        LITHIUM_CARB._300MG_CAPS       0.00      0.00      0.00         6
                    LITHIUM_CARBONATE_150MG_CAPS       0.00      0.00      0.00         5
                  LITHIUM_CARBONATE_Capsules_USP       0.00      0.00      0.00         4
                  LITHIUM_CARBONATE_ER_300MG_TAB       0.00      0.00      0.00         7
                          LITHIUM_CARB_300MG_TAB       0.00      0.00      0.00         3
                       LITHIUM_CARB_450MG_ER_TAB       0.00      0.00      0.00         3
                                  LIVALO_1MG_TAB       0.00      0.00      0.00         7
                                  LIVALO_2MG_TAB       0.00      0.00      0.00         4
                                  LIVALO_4MG_TAB       0.00      0.00      0.00         8
                                LODOSYN_25MG_TAB       0.00      0.00      0.00         8
                                       LOPRESSOR       0.00      0.00      0.00         2
                          LOTREL_5_20MG_CAPSULES       0.00      0.00      0.00         6
                              LOTREL_CAP_10-40MG       0.00      0.00      0.00         4
                               LOTREL_CAP_5-40MG       0.00      0.00      0.00         5
                                    LYRICA_150MG       0.00      0.00      0.00         4
                                     LYRICA_50MG       0.00      0.00      0.00         4
                                     LYRICA_75MG       0.00      0.00      0.00         8
                                LYRICA_CAP_100MG       0.00      0.00      0.00         8
                                LYRICA_CAP_200MG       0.00      0.00      0.00         2
                                LYRICA_CAP_225MG       0.00      0.00      0.00         3
                                LYRICA_CAP_300MG       0.00      0.00      0.00        11
                                 LYRICA_Capsules       0.04      0.50      0.07         2
                      MERCAPTOPURINE_Tablets_USP       0.00      0.00      0.00         2
                           METHERGINE_0.2MG_TABS       0.00      0.00      0.00         8
                         METHOTREXATE_2.5MG_TABS       0.00      0.00      0.00         7
                 MOEXIPRIL_HYDROCHLORIDE_Tablets       0.00      0.00      0.00         6
             MONTELUKAST_SODIUM_Chewable_Tablets       0.00      0.00      0.00         4
                                MULTAQ_TAB_400MG       0.00      0.00      0.00         7
                             MYCOBUTIN_CAP_150MG       0.00      0.00      0.00         5
              MYCOPHENOLATE_MOFETIL_Capsules_USP       0.00      0.00      0.00         3
                         MYCOPHENOLAT_500_MG_TAB       0.00      0.00      0.00         8
                              MYFORTIC_TAB_360MG       0.00      0.00      0.00         6
                            NAPROXEN_Tablets_USP       0.00      0.00      0.00        24
                           NARATRIPTAN_2.5MG_TAB       0.00      0.00      0.00         5
                          NIASPAN_1000MG_TABLETS       0.00      0.00      0.00         5
                                NIASPAN_ER_500MG       0.00      0.00      0.00         9
                            NITROSTAT_0.6_MG_TAB       0.00      0.00      0.00         4
                           NORVASC_10_MG_TABLETS       0.00      0.00      0.00         7
                 NOXAFIL_Delayed-Release_Tablets       0.00      0.00      0.00         4
                                        Naproxen       0.00      0.00      0.00         5
                                       Nitrostat       0.04      0.40      0.06         5
                               ONGLYZA_2.5MG_TAB       0.00      0.00      0.00         3
                           OXCARBAZEPINE_Tablets       0.02      0.43      0.04         7
                           PANTOPRAZOLE_20MG_TAB       0.00      0.00      0.00        11
                           PANTOPRAZOLE_40MG_TAB       0.00      0.00      0.00         7
 PANTOPRAZOLE_SODIUM_Delayed-Release_Tablets_USP       0.00      0.00      0.00        14
                                         PARNATE       0.00      0.00      0.00         6
                    PERINDOPRIL_ERBUMINE_Tablets       0.00      0.00      0.00         7
                              PREDNISONE_1MG_TAB       0.00      0.00      0.00         7
                          PREDNISONE_Tablets_USP       0.00      0.00      0.00         6
                                        PREMARIN       0.00      0.00      0.00         3
                           PREMARIN_Tablets,_USP       0.03      0.22      0.06         9
                PRISTIQ_Extended-Release_Tablets       0.00      0.00      0.00         2
                                PRISTIQ_TAB_50MG       0.00      0.00      0.00         9
                                PROMACTA_TABLETS       0.00      0.00      0.00        19
                         PROMETRIUM_Capsules_USP       0.00      0.00      0.00         1
                              Potassium_Chloride       0.00      0.00      0.00         6
                                      PredniSONE       0.03      0.28      0.06        18
                                       Procardia       0.00      0.00      0.00         4
                     QUETIAPINE_FUMARATE_Tablets       0.01      0.20      0.03        10
      RABEPRAZOLE_SODIUM_Delayed-Release_Tablets       0.00      0.00      0.00         3
                              RAMIPRIL_2.5MG_CAP       0.00      0.00      0.00         6
                                RAMIPRIL_5MG_CAP       0.00      0.00      0.00         6
                                RELPAX_40_MG_TAB       0.00      0.00      0.00         8
                                RELPAX_TAB_20_MG       0.00      0.00      0.00        11
                                          REQUIP       0.03      0.07      0.04        46
                                       REQUIP_XL       0.04      0.51      0.07        39
                           RILUTEK_50_MG_TABLETS       0.00      0.00      0.00         3
                                        Ramipril       0.00      0.00      0.00         8
                                     SANCTURA_XR       0.00      0.00      0.00         6
                              SIMCOR_TAB_1000-20       0.00      0.00      0.00         7
                                 SINEMET_Tablets       0.00      0.00      0.00         4
                               SINGULAIR_Tablets       0.00      0.00      0.00         5
                                 SOMA_250_MG_TAB       0.00      0.00      0.00         5
                                     STALEVO_100       0.00      0.00      0.00        11
                           STARLIX_120MG_TABLETS       0.00      0.00      0.00         4
                                STARLIX_60MG_TAB       0.00      0.00      0.00         6
                                  STRATTERA_10MG       0.00      0.00      0.00         6
                            STRATTERA_CAP_100_MG       0.00      0.00      0.00         5
                             STRATTERA_CAP_18_MG       0.00      0.00      0.00         3
                             STRATTERA_CAP_25_MG       0.00      0.00      0.00         6
                             STRATTERA_CAP_40_MG       0.00      0.00      0.00         7
                             STRATTERA_CAP_60_MG       0.00      0.00      0.00         9
                             STRATTERA_CAP_80_MG       0.00      0.00      0.00         8
                              STROMECTOL_3MG_TAB       0.00      0.00      0.00         8
                              SUSTIVA_TAB_600_MG       0.00      0.00      0.00         6
                        SYNTHROID_0.025MG_ORANGE       0.00      0.00      0.00         1
                       SYNTHROID_0.075_MG_VIOLET       0.00      0.00      0.00         3
                        SYNTHROID_0.088_MG_OLIVE       0.00      0.00      0.00         7
                         SYNTHROID_0.1_MG_YELLOW       0.00      0.00      0.00         8
                           SYNTHROID_0.2_MG_PINK       0.00      0.00      0.00         6
                           SYNTHROID_112_MCG_TAB       0.00      0.00      0.00         4
                           SYNTHROID_125_MCG_TAB       0.00      0.00      0.00        11
                           SYNTHROID_137_MCG_TAB       0.00      0.00      0.00         5
                           SYNTHROID_150_MCG_TAB       0.00      0.00      0.00         7
                           SYNTHROID_175_MCG_TAB       0.00      0.00      0.00         8
                           SYNTHROID_300_MCG_TAB       0.00      0.00      0.00         7
                          SYNTHROID_Tablets,_USP       0.00      0.00      0.00         5
                                  Simcor_Tablets       0.00      0.00      0.00         2
                          TAMIFLU_75_MG_CAPSULES       0.00      0.00      0.00         7
                                TAMIFLU_Capsules       0.00      0.00      0.00         3
                             TARKA_4_MG___240_MG       0.00      0.00      0.00         6
                                   TEGRETOL_-_XR       0.00      0.00      0.00         9
                         TEGRETOL_200_MG_TABLETS       0.00      0.00      0.00         6
                         TEKTURNA_150_MG_TABLETS       0.00      0.00      0.00        10
                         TEKTURNA_300_MG_TABLETS       0.00      0.00      0.00         2
                     TEKTURNA_HCT_TAB_150_-_12.5       0.00      0.00      0.00         6
 TELMISARTAN_and_HYDROCHLOROTHIAZIDE_Tablets_USP       0.00      0.00      0.00         4
                                  TEVETEN_600_MG       0.00      0.00      0.00         3
                     TEVETEN_HCT_TAB_600_-_25_MG       0.00      0.00      0.00         5
                       TICLOPIDINE_250MG_TABLETS       0.00      0.00      0.00         4
           TICLOPIDINE_HYDROCHLORIDE_Tablets_USP       0.00      0.00      0.00        11
                                         TIKOSYN       0.00      0.00      0.00         2
                              TINIDAZOLE_Tablets       0.00      0.00      0.00         3
                    TOLTERODINE_TARTRATE_Tablets       0.00      0.00      0.00         6
                              TOPIRAMATE_Tablets       0.00      0.00      0.00         5
                                       TORSEMIDE       0.00      0.00      0.00         3
                        TRICOR_48_MG_YELLOW_TABS       0.00      0.00      0.00         5
                       TRILEPTAL_150MG_GREY_TABS       0.00      0.00      0.00         6
                     TRILEPTAL_300MG_YELLOW_TABS       0.00      0.00      0.00         4
                       TRILEPTAL_600MG_PINK_TABS       0.00      0.00      0.00         6
                             TRILIPIX_CAP_135_MG       0.00      0.00      0.00         8
                               TRILIPIX_CAP_45MG       0.00      0.00      0.00         3
                                Tamiflu_Capsules       0.00      0.00      0.00         4
                      TriCor_FENOFIBRATE_Tablets       0.00      0.00      0.00         6
                           ULTRACET_37.5-325_TAB       0.00      0.00      0.00         4
                                URIBEL_118MG_CAP       0.00      0.00      0.00         3
                                 UROXATRAL_10_MG       0.00      0.00      0.00         9
                          VALTURNA_TAB_150_-_160       0.00      0.00      0.00         8
                          VALTURNA_TAB_300_-_320       0.00      0.00      0.00         8
               VENLAFAXINE_HYDROCHLORIDE_Tablets       0.00      0.00      0.00         7
                                   VIAGRA_100_MG       0.00      0.00      0.00         7
                                    VIAGRA_25_MG       0.00      0.00      0.00         8
                                    VIAGRA_50_MG       0.00      0.00      0.00         8
                              VICODIN_ES_TABLETS       0.00      0.00      0.00         7
                       Venlafaxine_Hydrochloride       0.00      0.00      0.00         6
                                XANAX_0.5_MG_TAB       0.00      0.00      0.00         5
                                   XYZAL_TAB_5MG       0.00      0.00      0.00         8
                   Xeloda_(CAPECITABINE)_Tablets       0.00      0.00      0.00         5
                                     ZOCOR_40_MG       0.00      0.00      0.00         9
                               ZOLOFT_TAB_100_MG       0.00      0.00      0.00         6
                                 ZOLOFT_TAB_25MG       0.00      0.00      0.00         4
                       ZOLPIDEM_TARTRATE_Tablets       0.00      0.00      0.00         7
                       ZYVOX_(LINEZOLID)_Tablets       0.00      0.00      0.00         3
                                  Zoloft_Tablets       0.00      0.00      0.00         2
                                 onglyza_Tablets       0.00      0.00      0.00         2
                  traMADOL_Hydrochloride_Tablets       0.00      0.00      0.00         7
              traMADOL_Hydrochloride_Tablets_USP       0.02      0.17      0.04         6

                                        accuracy                           0.03      2234
                                       macro avg       0.00      0.01      0.00      2234
                                    weighted avg       0.00      0.03      0.01      2234

---

## Final Thoughts

This took way more time than expected because of how bad the raw data was and how fragile Spark is when it comes to memory and long-running jobs. Still, I got something working. I explained all of it in the report, including all the things I had to redo and what I’d improve if I kept going.
