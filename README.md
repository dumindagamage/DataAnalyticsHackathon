# Car Price Prediction Multiple Linear Regression - Group Project (Data Wranglers)
**Group Members:**
- Data Architect: Michael
- Data Analyst: Collins 
- Project Manager: Duminda 

**Car Price Prediction Multiple Linear Regression** is an all-in-one data analysis tool that simplifies data exploration, processing, and visualisation. It supports a wide range of data formats and enables efficient workflows for all types of data scientists. 

## Dataset Content
The dataset we used is from [Kaggle Car Price Prediction](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction/data) the data is titled `CarPrice_Assignment.csv`

## Business Requirements
The business requires a predictive car pricing model that uses available independent variables to reveal how each factor influences price, enabling management to adjust design and strategy to meet target price levels while also gaining insights into pricing dynamics for new markets.

## Hypothesis And How To Validate
## Hypothesis 1.1: Bar Chart 
Car brand affect average car price.

### Result: 
The more expensive car brands have higher average prices in the market.

## Hypothesis 1.2: Bar Chart
H1.2: Car Body Type influences the price 

### Result:
Hardtops and convertibles have a significantly higher average price than sedans, wagons and hatchbacks.
## Hypothesis 2: Scatter Plot and Heatmap
Engine size , Car Space and Fuel Efficiency has positive correlations with price.

### Result:

## Hypothesis 3: Bar Chart
Car prices vary significantly by manufacturing region.

### Result: 
European and North American markets are significantly higher in average prices compared to Asia.

## Project Plan
* Process the dataset and assign roles (Data Architect, Data Analyst, Project Manager) 
* Clean the data (check for errors, duplicates, missing values, etc.)
* Load and create interactive visualisations and dashboards.
* Present information using the visualisations and dashboards.

## The Rationale To Map The Business Requirements To Data Visualations
* The first visualisation is a bar chart (Descriptive) which displays the average price by brand that shows the benefit of selling high range of luxury brands. 
* The second visualisation is a bart chart (Descriptive) which shows the average price by body type (hardtop, convertible, sedan, wagon, hactchback) where it shows the two main types that provides higher sales (hardtops and convertibles).
* The third visualisation is a heatmap (correlation) which shows the engine performance, drives the price as it has a correalation of 0.806. The fuel efficiency negatively impacts the price as the correalation is -0.7. Large vehicles have larger prices that also has a correalation of 0.6 that proves the statement that larger vehicles cost more.
* The fourth visualisation is a scatter plot (correlation) which is used to additionally prove the previous statement that larger vehicles cost more.
* The fifth visualisation is a bar chart (geographical) that shows the drastic difference in sales in Europe and North American where it is clear that the markets in these regions are far more benefitical. 

## Analysis Techniques Used
Prepare, validate and transform the data using Jupyter Notebooks.
Visualisations: Box plots, Bar Charts, Descriptive Statistics, Heat Maps.
PowerBI and Streamlit to create dashboards.

## Ethical Considerations
* No sensitive data is including in the dataset and is public
* Bias could be considered due to the choice of markets but that is beyond our power 

## Business Justifications
We wanted to generate a visual around showing the prices over time. However, the dataset does not provide us with this information. We would've calculcated the sales of the car brands and models since their manufacture in the target regions but this was not supplied to us so it was not applicabl for us to complete this research and would've taken too much time and resources. 

* Luxury Market Representation: High-priced outliers (Porsche, Jaguar, BMW) represent the luxury segment your company needs to understand
* Market Reality: These are genuine market prices, not data errors
* Strategic Value: Understanding premium segments helps positioning strategy
* Small Dataset Impact: Removing outliers would reduce an already small dataset significantly

## Planning
To ensure that everyone was on track we created a project board so that we could all see our tasks and make sure they were being updated (to do, in progress and completed) https://github.com/users/dumindagamage/projects/4/views/1 
 
 ## Main Data Analysis Libraries
* Pandas - For data manipulation
* Numpy - For data manipulation
* Streamlit - For an interactive web appliaction to display findings
* Plotly - For interactive visualisations
* Matplotlib - For plot visualisations
* Statsmodels - For statistical modeling and hypothesis testing
* (AI) ChatGPT and CoPilot for general data error fixes

## Credits 
- Dataset used in this project from Kaggle
- AI support throughout with any errors and to generate ideas for hypotheses
- Peer support and feedback to form higher standard content and fix each other with any issues

## Acknowlegdements
* Very thankful of one another in the group for supporting and completing the project together and keeping clear communication throughout.
* Special thanks to our work coaches: Emma Lamont, Spencer Barriball and Mark Briscoe for the support and knowledge they have supplied us with so far! 