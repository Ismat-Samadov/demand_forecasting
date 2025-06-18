```markdown
# 🧠 M5 Forecasting – Accuracy Dataset Documentation

This repository contains the official dataset files for the **M5 Forecasting - Accuracy** competition hosted on Kaggle, in which participants forecast the daily unit sales of Walmart products across three US states for a 28-day forecast horizon.

The competition aims to advance both the **theory and practice of demand forecasting** by leveraging historical sales data, product metadata, calendar events, and price changes.

> **Link to competition**: [https://www.kaggle.com/competitions/m5-forecasting-accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy)

---

## 🗂️ Repository Structure

```

.
├── data
│   ├── calendar.csv                  # Calendar information including holidays and events
│   ├── sales\_train\_evaluation.csv   # Sales data from day 1 to day 1941 (for evaluation)
│   ├── sales\_train\_validation.csv   # Sales data from day 1 to day 1913 (for validation)
│   ├── sample\_submission.csv        # Sample format of the submission file
│   └── sell\_prices.csv              # Price data per item-store-week
├── LICENSE
└── README.md                        # Dataset documentation (this file)

```

---

## 📅 Dataset Overview

| Name                        | Description                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| **calendar.csv**            | Maps the date index (`d_1` to `d_1969`) to real-world dates and events      |
| **sales_train_validation.csv** | Unit sales per item/store for training (validation) up to `d_1913`       |
| **sales_train_evaluation.csv** | Extended version for final evaluation including sales up to `d_1941`   |
| **sell_prices.csv**         | Weekly price data per item/store                                           |
| **sample_submission.csv**   | Required format for model submission (includes `F1` to `F28`)              |

---

## 📘 Column Descriptions

### 📄 `calendar.csv`
This file provides date-related information that supports modeling of seasonality, holidays, and special events.

| Column Name     | Description                                              |
|-----------------|----------------------------------------------------------|
| `date`          | Date in `yyyy-mm-dd` format                              |
| `wm_yr_wk`      | Walmart’s unique identifier for the retail week          |
| `weekday`       | Day of the week (Monday, Tuesday, ...)                   |
| `wday`          | Numeric day of the week (1 to 7)                          |
| `month`         | Month number (1–12)                                       |
| `year`          | Year number                                              |
| `event_name_1`  | First event name if any (e.g., Super Bowl, Christmas)    |
| `event_type_1`  | Type of first event (Holiday, Sporting, etc.)            |
| `event_name_2`  | Second event name (if applicable)                        |
| `event_type_2`  | Type of second event                                     |
| `snap_CA`       | SNAP (food assistance) indicator in California           |
| `snap_TX`       | SNAP indicator in Texas                                  |
| `snap_WI`       | SNAP indicator in Wisconsin                              |

---

### 📄 `sales_train_validation.csv` & `sales_train_evaluation.csv`
These files contain daily unit sales data.

| Column Name     | Description                                              |
|-----------------|----------------------------------------------------------|
| `id`            | Unique row identifier: `item_id` + `store_id` + `_validation`/`_evaluation` |
| `item_id`       | Unique product identifier                                |
| `dept_id`       | Department ID                                            |
| `cat_id`        | Category ID                                              |
| `store_id`      | Store ID                                                 |
| `state_id`      | US state ID (`CA`, `TX`, `WI`)                           |
| `d_1` to `d_1913` or `d_1941` | Unit sales for each day                    |

> ⚠️ Each row represents an individual item-store combination.

---

### 📄 `sell_prices.csv`
Provides the actual selling prices for each item at each store and week.

| Column Name     | Description                                              |
|-----------------|----------------------------------------------------------|
| `store_id`      | Store ID                                                 |
| `item_id`       | Item ID                                                  |
| `wm_yr_wk`      | Retail week number (matches `calendar.csv`)             |
| `sell_price`    | Price of the item in the given week                     |

> This file enables modeling of price sensitivity and promotional effects.

---

### 📄 `sample_submission.csv`
Used to format final submission.

| Column Name     | Description                                              |
|-----------------|----------------------------------------------------------|
| `id`            | Same format as in training data (`_validation` or `_evaluation`) |
| `F1`–`F28`      | Forecasted sales for 28 consecutive days                 |

---

## 📈 Forecasting Task

- Predict the unit sales for the next **28 days** (`F1` to `F28`) for each item/store combination.
- Your forecasts must be aligned with the exact `id` format and structure as in `sample_submission.csv`.

---

## 🧮 Evaluation Metric

The competition uses **Weighted Root Mean Squared Scaled Error (WRMSSE)**:

- **Weighted**: Different products contribute differently to the final score.
- **Scaled**: RMSE is normalized by the scale of past demand to penalize errors proportionally.
- The exact computation is documented in the M5 Participants Guide.

---

## 📅 Timeline (Historical)

- **Start**: March 3, 2020  
- **Final Submission**: June 30, 2020  
- **Public Leaderboard Release**: June 1, 2020  
- **Prizes**: $45,000 total, including a student award

---

## 📖 Citation

> Addison Howard, inversion, Spyros Makridakis, and Vangelis.  
> **M5 Forecasting - Accuracy**. [https://kaggle.com/competitions/m5-forecasting-accuracy](https://kaggle.com/competitions/m5-forecasting-accuracy), 2020.

---

## 🏆 Competition Partners

- University of Nicosia
- Makridakis Open Forecasting Center (MOFC)
- National Technical University of Athens
- INSEAD
- Google
- Uber
- International Institute of Forecasters (IIF)

---

## 📬 Contact & License

This dataset is provided under Kaggle's competition terms. For academic use, cite the competition and link to the dataset.

---

```
