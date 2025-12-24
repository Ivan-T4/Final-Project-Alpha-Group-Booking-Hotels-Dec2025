# Final-Project-Alpha-Group-Booking-Hotels-Dec2025

Final Project about Hotel Booking Demand 
Created by Ivan Taufiqurrahman github.com/Ivan-T4 and M Fawwaz Firjatullah Nursyahdian github.com/

dataset source: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand/data

---

**Link Video:**
---
**Link Presentation:**
---
**Link Application:** https://final-project-alpha-group-booking-hotels-dec2025-ivan-fawwaz.streamlit.app/
---
**Link Tableau:** https://public.tableau.com/app/profile/fawwaz.firjatullah/viz/final_17660709970440/Dashboard?publish=yes
---

## Context

During 2015–2017, Portugal experienced a major tourism surge, with international arrivals rising significantly and reaching 15.4 million visitors in 2017. This growth brought substantial economic benefits as tourism spending increased sharply between 2016 and 2017, reflecting strong demand from international travelers. As hotels enjoyed higher reservation volumes, they also faced new operational pressures from fluctuating seasonal demand and greater dependence on online booking channels. These conditions shaped a rapidly expanding yet increasingly unpredictable hospitality environment. 

Amid this boom, hotels began seeing a rise in last minute cancellations, which created challenges in revenue forecasting, room inventory planning, and operational efficiency. Even with full pipelines of bookings, many hotels struggled because cancellations frequently left rooms unsold despite the high demand. The Hotel Booking Demand Kaggle dataset captures these real behaviors from a city and resort hotel in Portugal during this period, reflecting the industry’s struggle to manage growing cancellation patterns. In this context, machine learning becomes an essential tool, enabling hotels to predict cancellation risks and make proactive, data driven decisions to protect revenue during a time of rapid tourism growth.

reference: 
- https://www.ceicdata.com/en/portugal/tourism-statistics

- https://www.ine.pt/xportal/xmain?xpid=INE&xpgid=ine_publicacoes&PUBLICACOEStipo=ea&PUBLICACOEScoleccao=5586641
---

## Problem Statement

Despite the significant growth of Portugal’s tourism sector between 2015–2017, hotels increasingly struggled with unpredictable booking cancellations that led to revenue loss, inaccurate demand forecasting, and inefficient room allocation. As booking behaviors shifted toward online platforms, cancellation rates rose and became harder to anticipate. This growing unpredictability created operational and financial challenges, highlighting the need for a better understanding of the factors that drive guest cancellations.

---

## Solution

Because the target is classification, by applying machine learning using model like XGBoostClassifiaction, LGBMClassification, and other Classification models, hotels can better anticipate which bookings are likely to be canceled, allowing them to reduce the operational and financial costs caused by false negatives (missed high‑risk cancellations) and false positives (incorrectly flagged reservations). This data‑driven approach helps hotels make more accurate decisions about forecasting, overbooking strategies, and resource allocation, ultimately minimizing revenue loss and improving occupancy stability.

---

## Goals

- Optimize room availability and revenue by identifying bookings likely to be canceled before they occur.

- Understand guest behavior patterns that lead to cancellations to support better marketing and pricing decisions.

- To get the highest accuracy percentage for every customer's behaviour and details using the best machine learning model.

- Minimize financial loss and operational inefficiency by turning predictive insights into proactive booking management strategies.

- To create a learning benchmark that can help others reflect on how such models might be adapted or improved for different hotels or resorts market.

---

## Stakeholder

| **Stakeholders** | **Purpose** |
| --- | --- |
|Hotel Managers| Use predictions from machine learning to plan staffing and room availability more efficeintly|
|Revenue Management Teams| Adjusting price and promotions to reduce losses from cancellations|
|Booking Coordinators| Identify high risk bookings early and follow up with guests to confirm or rebook|
|Marketing / Customer Relations| make retention campaigns for guests likely to cancel|

---

## Analytical Approach
The project follows a structured machine learning pipeline:

1. **Data Cleaning** → handling missing values, preparing numerical and categorical features, handling outliers.

2. **Data Analysis** → analyzing cleaned data to get insights.

3. **Feature Transformation** → scaling, encoding.

4. **Model Development** → training multiple algorithms, both baseline and advanced.

5. **Model Tuning** → optimizing hyperparameters to improve performance.

6. **Model Interpretation** → using feature importance and explainable AI techniques to understand predictions.

**Additional Supporting Tools for this project**

1. **Data Visualization with Tableau** → Creating visualizations that more compeling, insightfull, and easy to understand for the stakeholders.

2. **Streamlit** → Creating an program or application that can predict cancellation probability of new customer through their booking behaviour or detail information.

---

**Confusion Matrix** 

| **Error Type** | **Intepretation** | **Business Impact** | **Cost** |
| --- | --- | --- | --- |
|False Positive (FP)|The model predicts a booking will be canceled, but in reality, the guest shows up.|❌ Customer dissatisfaction, brand damage, and compensation costs.|Possible overbooking → must relocate or compensate guest. $172 (see before conclusion in notebook) |
|False Negative (FN)|The model predicts a booking will not be canceled, but it does cancel.|❌ Lost revenue, especially if not rebooked in time.|Room stays empty → direct lost room revenue. (mean adr * mean total nights) = $343 (see before conclusion in notebook) |


in this case we see that the impact of empty room cost double than customer dissatisfaction and compensate for other room in same hotel or better
---
$$
Fβ​=(1+β^2)⋅((Precision⋅Recall)/((β^2⋅Precision)+Recall))​)
 $$

with β = 2, then

$$
F2=5⋅((Precision⋅Recall)/((4⋅Precision)+Recall))
$$

## Model Evaluation

To fairly assess the models, multiple evaluation metrics are used:

| **Metric**               | **Purpose**                      | **Reason**                                                                                                                                                        |
| :------------------- | :--------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `F2`          | Model tuning & final scoring | To emphasize that the impact of False Negative (FN) is more costly than False Positive (FP) |
| `Confusion Matrix` | Business interpretation      | To visualize classification results and understand FP/FN trade-offs. |

Together, these metrics provide a balanced view of performance, from both absolute and proportional perspectives.

---

## Additional Information

- This Dataset depict booking registration each customer made to hotels or resort
- each booking does not necessarily represent 1 people as customer, and can be more
- The dataset is not proportional
- Most categorical feature are Nominal or Binary

## Credits
Dataset posted by Jesse Mostipak in Kaggle and originally from the article Hotel Booking Demand Datasets, written by Nuno Antonio, Ana Almeida, and Luis Nunes for Data in Brief, Volume 22, February 2019.

The data was downloaded and cleaned by Thomas Mock and Antoine Bichat for #TidyTuesday during the week of February 11th, 2020.

---
## Limitation in This Project

- this model is not suitable for different demography or different hotel/resort in different country with different booking cultures

- in the current state after pandemic covid-19 customers might have different behaviour when they booking a room in hotel or resort

- Channel to booking a room or resort change from 2017 like using new platform that doesnt exist in 2017

- Impact on social media in current era has significantly increased so the cost of brand damage, customer dissatisfaction that ranting in social media might more costly than empty room which in this case FP > FN so the confusion matrix might differ and need adjustment.

- using ROS can inflate minority class and cause overfitting

- Hundreds of features because of transforming many features using BinaryEncoding and OneHotEncoding makes intrepretation harder especially for country_full

---
## Conclusion

- About 1 in 3 hotel bookings end up being canceled — a major operational and financial challenge.

- City Hotels, Online Travel Agent (OTA) bookings, and long lead-time reservations have the highest cancellation risk.

- Guests who make special requests or require parking are more reliable and less likely to cancel.

- Seasonal peaks (April–August) show higher cancellation rates, despite high booking demand.

- The Tuned XGBoost model (F₂ = 0.76) successfully predicts high-risk cancellations with strong accuracy.

- Using the ML model, the hotel could reduce total financial loss from $1.33M to $650K, saving approximately $677,220 in potential revenue.

---

## **Actionable Recommendation for Hotel/Resort Managers**
- Adopt ML-based cancellation monitoring
Implement the predictive model in daily booking operations to identify and manage high-risk bookings early — this alone could save over $670K annually.

- Refine deposit and cancellation policies
Require partial deposits (25–40%) or stricter cancellation windows for OTA and long-lead bookings to minimize speculative reservations.

- Prioritize reliable guest segments
Promote Direct, Corporate, and Group bookings through loyalty points, discounts, or exclusive perks to stabilize revenue.

- Plan resources by seasonality
During high-demand months, prepare for higher cancellations by using controlled overbooking, flexible staffing, and targeted re-marketing.

- Leverage guest behavior signals
Guests with special requests or parking needs show high commitment — offer tailored upsells or rewards to retain and convert these loyal segments.
---
## Important Note

Untuk penjalanan streamlit/program perlu pip install -r runtime.txt dan pip install -r requirements.txt sebelum streamlit run Booking_cancellations_v2.py.
Pastikan juga semua file ada pada satu PATH yang sama untuk berjalan lancar instalasi programnya.
