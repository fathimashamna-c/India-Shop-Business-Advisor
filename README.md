

# 🏪 India Shop Business Advisor

An ML-powered business recommendation application that helps aspiring entrepreneurs identify and compare suitable shop business opportunities across India based on investment budget, monthly rent, local market conditions, competition, and estimated demand.

## 📌 Project Overview

Starting a new business requires careful planning and an understanding of the local market. Choosing the wrong business location or investing in a business with insufficient demand can lead to financial losses.

The **India Shop Business Advisor** helps users explore potential business opportunities by analyzing location-related information, investment budget, rental expenses, competition, and estimated customer demand.

The application combines data analysis, machine learning, and an interactive Streamlit interface to support business planning and decision-making.

## 🎯 Objectives

- Recommend suitable shop business categories based on investment budget.
- Analyze business opportunities across Indian states, cities, and neighbourhoods.
- Estimate potential demand using a Machine Learning model.
- Consider monthly shop rent and competitor information.
- Compare business opportunities and estimated financial outcomes.
- Support entrepreneurs in making data-informed business decisions.

## ✨ Features

- 📍 Location-based business analysis
- 🏙️ State, city, and neighbourhood selection
- 💰 Investment budget analysis
- 🏢 Monthly rental cost consideration
- 🛍️ Business category comparison
- 🏪 Competitor analysis
- 🤖 Machine Learning-based demand prediction
- 📊 Estimated transactions, revenue, expenses, and profit
- 🌐 Interactive web application

## 🧠 Machine Learning

The project uses a **Random Forest Regressor** to estimate monthly demand transactions for different shop business categories.

### Model Inputs

- Neighbourhood
- Shop Category
- Population
- Number of Competitors
- Monthly Rent (INR)

### Model Output

- Estimated Monthly Demand Transactions

The model is trained using business-related data and uses the learned patterns to generate demand estimates.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computations |
| Scikit-learn | Machine Learning model |
| Random Forest Regressor | Demand prediction |
| Joblib | Model saving and loading |
| Streamlit | Interactive web application |
| OpenStreetMap (OSM) | Location and business information |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |

## 🏗️ Project Structure

```text
India-Shop-Business-Advisor/
│
├── app.py
├── requirements.txt
├── README.md
│
├── Data/
│   ├── Raw/
│   └── Processed/
│
├── outputs/
│   └── shop_demand_model.joblib
│
├── analyze_data.py
├── train_model.py
├── predict_demand.py
├── visualize_data.py
├── download_osm_data.py
└── business_comparison.py
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/fathimashamna-c/India-Shop-Business-Advisor.git
```

### 2. Navigate to the project folder

```bash
cd India-Shop-Business-Advisor
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📈 Application Workflow

1. User selects a state, city, and neighbourhood.
2. User enters the investment budget and monthly shop rent.
3. The application collects or uses available location and market information.
4. Business categories are evaluated.
5. The Machine Learning model estimates monthly demand.
6. Financial estimates are calculated.
7. Business opportunities are compared and presented through the interactive dashboard.

## 🌍 Data Sources

- Business and location information from OpenStreetMap (OSM), where available.
- Project datasets used for training and analysis.
- Location-related information and assumptions used in the application.



## 🚀 Deployment

The application is deployed using **Streamlit Community Cloud**.

Users can access the interactive application through the deployed web link without installing Python locally.





⭐ If you find this project interesting, feel free to explore it!
