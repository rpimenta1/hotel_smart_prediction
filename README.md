# Hotel Booking Cancellation Prediction

## 🏨 Why Do Hotel Bookings Get Canceled — and How Can We Predict It?

In recent years, **online hotel booking platforms** have completely transformed how people reserve their stays. With just a few clicks, travelers can book, reschedule, or cancel a reservation — all in real time. But this convenience comes with a cost.

Today, a **large number of bookings end up being canceled**, either intentionally by customers or through no-shows. The reasons? They vary — from last-minute changes of plans to simple schedule conflicts. And with the widespread availability of **free or low-cost cancellation policies**, guests are empowered to make flexible decisions. Great for them — but potentially risky for hotel operations.

While guests enjoy the freedom to change their minds, **hotels face the consequences**: empty rooms and increase of costs. Each canceled booking can cost a hotel **an average of R$3,500**, and with **50,000 reservations expected next year**, that risk adds up fast.

---

## 💡 The Challenge: Turning Data into Smart Decisions

You are a **Data Scientist at HotelSmart**, a fictional hospitality company where the CEO is alarmed by the rising cancellation rate and its financial toll. The mission? Build a **Machine Learning model** capable of accurately predicting which bookings are likely to be canceled.

By forecasting cancellations ahead of time, HotelSmart can:

- Take **proactive measures** to mitigate cost rise  
- **Optimize room occupancy** by adjusting inventory strategies  
- Improve planning and **boost profitability**

This project is not only about about predictive accuracy but also, it’s about using data to drive smarter, more resilient business decisions

---

## 📊 Technologies Used

- Python
- Pandas & NumPy
- Scikit-Learn
- Flask API
- Pickle for model serialization

---

## Data Science Techniques
- Attribute Selection (Boruta)
- Fine Tuning (Grid Search)
- Data Cleaning
- Scaling (RobustScaler)
- Encoding (Label Encoder)
- Univariate analysis
- Bivariate analysis
- Multivariable Analysis
- Elbow Analysis

## 🚀 Next Steps

- Deploy the model to production using Koyeb or a similar cloud platform
- Monitor performance and retrain regularly
- Integrate predictions into hotel management systems for real-time decision-making

---
## Notes
- Although GridSearchCV was initially used during experimentation to identify the optimal hyperparameters for the model, it was not included in the final production pipeline for a few key reasons:

    Resource Efficiency: Grid search is computationally intensive. Running it during each deployment would significantly increase the memory and processing requirements of the model.

    Deployment Constraints: The project is deployed on Koyeb's free tier, which has limited resources. Including the full grid search process would exceed those limitations, potentially causing service interruptions or requiring a paid subscription.

## 📫 Contact

For questions or feedback, feel free to reach out via GitHub or email.