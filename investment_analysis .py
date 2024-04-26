#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


# In[3]:


df_finance = pd.read_csv(r'C:\Users\Melikzade\Desktop\SQL Course Materials\Finance_data.csv')
df_data = pd.read_csv(r'C:\Users\Melikzade\Desktop\SQL Course Materials\Original_data.csv')


# In[5]:


df_finance.head()


# In[7]:


df_finance.describe()


# In[13]:


df_data.head()


# In[8]:


df_data.describe()


# In[12]:


# Gender distribution
gender_counts = df_data['GENDER'].value_counts()
gender_percentage = gender_counts / gender_counts.sum() * 100

# Age distribution
age_mean = df_data['AGE'].mean()
age_median = df_data['AGE'].median()
age_mode = df_data['AGE'].mode()[0]

# Visualization: Gender distribution
plt.figure(figsize=(8, 6))
plt.pie(gender_percentage, labels=gender_percentage.index, autopct='%1.1f%%', startangle=140)
plt.title('Gender Distribution of Respondents')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.show()

# Visualization: Age distribution
plt.figure(figsize=(8, 6))
plt.hist(df_data['AGE'], bins=10, color='skyblue', edgecolor='black')
plt.axvline(age_mean, color='red', linestyle='--', label=f'Mean: {age_mean:.2f}')
plt.axvline(age_median, color='green', linestyle='--', label=f'Median: {age_median}')
plt.axvline(age_mode, color='orange', linestyle='--', label=f'Mode: {age_mode}')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution of Respondents')
plt.legend()
plt.show()



# In[9]:


import matplotlib.pyplot as plt

# Investment options
investment_options = ['Mutual Funds', 'Equity Market', 'Debentures', 'Government Bonds', 
                      'Fixed Deposits', 'Public Provident Fund', 'Gold']

# Average preference rankings
average_rankings = [2.55, 3.475, 5.75, 4.65, 3.575, 2.025, 5.975]  # Taken from the provided data

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(investment_options, average_rankings, color='skyblue')
plt.xlabel('Average Preference Ranking')
plt.ylabel('Investment Options')
plt.title('Average Preference Ranking for Investment Options')
plt.gca().invert_yaxis()  # Invert y-axis to display highest preference at the top
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()


# In[20]:


# Rename the specified columns
df_data.rename(columns={
    'What do you think are the best options for investing your money? (Rank in order of preference) [Mutual Funds]': 'Mutual Funds',
    'What do you think are the best options for investing your money? (Rank in order of preference) [Equity Market]': 'Equity Market',
    'What do you think are the best options for investing your money? (Rank in order of preference) [Debentures]': 'Debentures',
    'What do you think are the best options for investing your money? (Rank in order of preference) [Government Bonds]': 'Government Bonds',
    'What do you think are the best options for investing your money? (Rank in order of preference) [Fixed Deposits]': 'Fixed Deposits',
    'What do you think are the best options for investing your money? (Rank in order of preference) [Public Provident Fund]': 'Public Provident Fund',
    'What do you think are the best options for investing your money? (Rank in order of preference) [Gold]': 'Gold'
}, inplace=True)

# Print the updated column names
print(df_data.columns)


# In[21]:


# Define age groups
age_bins = [0, 25, 30, float('inf')]
age_labels = ['<25', '25-30', '>30']

# Create a new column for age groups
df_data['Age Group'] = pd.cut(df_data['AGE'], bins=age_bins, labels=age_labels, right=False)

# Convert preference ranking columns to numeric type
preference_columns = df_data.columns[3:10]  # Columns containing preference rankings
df_data[preference_columns] = df_data[preference_columns].apply(pd.to_numeric, errors='coerce')

# Calculate average preference rankings for each investment option within each age group
avg_rankings_by_age_group = df_data.groupby('Age Group')[preference_columns].mean()

# Plotting
plt.figure(figsize=(12, 8))
avg_rankings_by_age_group.plot(kind='bar', colormap='viridis', figsize=(12, 8))
plt.title('Average Preference Rankings for Investment Options by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Average Preference Ranking')
plt.xticks(rotation=0)
plt.legend(title='Investment Option')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# In[23]:


# Extract responses from the column
factors_column = df_data['What are the factors considered by you while investing in any instrument?']

# Split responses into individual factors and create a list of all factors
all_factors = []
for response in factors_column:
    factors = response.split(', ')
    all_factors.extend(factors)

# Count the frequency of each factor
factor_counts = pd.Series(all_factors).value_counts()

# Calculate relative importance of each factor
total_responses = len(factors_column)
relative_importance = factor_counts / total_responses

# Identify the most influential factors
most_influential_factors = relative_importance.nlargest(5)

# Plotting
plt.figure(figsize=(12, 8))
most_influential_factors.plot(kind='bar', color='skyblue')
plt.title('Most Influential Factors Considered by Respondents')
plt.xlabel('Factors')
plt.ylabel('Relative Importance')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print("Most influential factors and their relative importance:")
print(most_influential_factors)


# In[26]:


# Extract responses from the columns
investment_objective_column = df_data['What is your investment objective?']
investment_purpose_column = df_data['What is your purpose behind investment?']

# Count the frequency of each investment objective and purpose
objective_counts = investment_objective_column.value_counts()
purpose_counts = investment_purpose_column.value_counts()

# Plotting
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
objective_counts.plot(kind='bar', color='skyblue')
plt.title('Investment Objectives of Respondents')
plt.xlabel('Investment Objective')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.subplot(2, 1, 2)
purpose_counts.plot(kind='bar', color='lightgreen')
plt.title('Purposes behind Investment Activities')
plt.xlabel('Purpose')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


# In[54]:


# Get the column names
column_names = df_finance.columns

# Print the column names
print(column_names)


# In[27]:


# Count the frequency of each investment objective
objective_counts = df_data['What is your investment objective?'].value_counts()

# Print the top 5 most common investment objectives
print("Top 5 Investment Objectives:")
print(objective_counts.head(5))

# Count the frequency of each investment purpose
purpose_counts = df_data['What is your purpose behind investment?'].value_counts()

# Print the top 5 most common investment purposes
print("\nTop 5 Investment Purposes:")
print(purpose_counts.head(5))


# In[32]:


# Extract responses from the columns
investment_duration_column = df_data['How long do you prefer to keep your money in any investment instrument?']
monitoring_frequency_column = df_data['How often do you monitor your investment?']

# Explore the distribution of preferred investment duration and monitoring frequency
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
investment_duration_column.value_counts().plot(kind='bar', color='skyblue')
plt.title('Preferred Investment Duration')
plt.xlabel('Duration')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.subplot(1, 2, 2)
monitoring_frequency_column.value_counts().plot(kind='bar', color='lightgreen')
plt.title('Monitoring Frequency')
plt.xlabel('Frequency')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()



# In[44]:


# Preprocess the data: remove '%' symbol and convert to numeric if applicable
for column in investment_returns_data.columns:
    if column != 'How much return do you expect from any investment instrument?':
        # Check if the column contains strings
        if investment_returns_data[column].dtype == 'object':
            investment_returns_data[column] = investment_returns_data[column].str.rstrip('%').astype(float)
        # Check if the column contains numeric values
        elif pd.api.types.is_numeric_dtype(investment_returns_data[column]):
            # Do nothing, it's already numeric
            pass
        else:
            print(f"Column '{column}' contains non-numeric data that cannot be processed.")


# In[43]:


# Get the column names
column_names = investment_returns_data.columns

# Print the column names
print(column_names)


# In[49]:


# Preprocess the data: remove '%' symbol and convert to numeric if applicable
for column in investment_returns_data.columns:
    if column != 'How much return do you expect from any investment instrument?':
        if investment_returns_data[column].dtype == 'object':
            investment_returns_data[column] = investment_returns_data[column].str.replace(r'\D+', '').astype(float)
        
        elif pd.api.types.is_numeric_dtype(investment_returns_data[column]):
            pass
        else:
            print(f"Column '{column}' contains non-numeric data that cannot be processed.")

# Extract responses from the columns related to investment returns and investment avenues
investment_returns_columns = ['How much return do you expect from any investment instrument?',
                              'Mutual Funds',
                              'Equity Market',
                              'Debentures',
                              'Government Bonds',
                              'Fixed Deposits',
                              'Public Provident Fund',
                              'Gold']
investment_returns_data = df_data[investment_returns_columns]

# Replace any non-numeric characters with an empty string
investment_returns_data = investment_returns_data.replace(r'\D+', '', regex=True)

# Convert the string values to numeric data types
investment_returns_data = investment_returns_data.astype(float)

# Analyze the distribution of respondents' expectations regarding investment returns for each investment avenue
plt.figure(figsize=(12, 8))

for column in investment_returns_columns[1:]:
    plt.hist(investment_returns_data[column], bins=10, alpha=0.5, label=column)

plt.title('Distribution of Expected Investment Returns Across Different Investment Avenues')
plt.xlabel('Expected Returns')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Compare the average expected returns across different investment avenues
average_returns = investment_returns_data.mean()
average_returns.plot(kind='bar', color='skyblue', figsize=(10, 6))
plt.title('Average Expected Returns Across Different Investment Avenues')
plt.xlabel('Investment Avenues')
plt.ylabel('Average Expected Returns')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[55]:


import seaborn as sns

# Create cross-tabulation table
investment_savings_table = pd.crosstab(df_finance['Avenue'], df_finance['What are your savings objectives?'])

# Calculate percentages for each cell
investment_savings_table_pct = investment_savings_table.apply(lambda x: x / x.sum() * 100, axis=1)

# Create heatmap
sns.heatmap(investment_savings_table_pct, annot=True, cmap="YlGnBu")
plt.title("Relationship Between Preferred Investment and Savings Objectives (Percentage)")
plt.xlabel("Preferred Investment Avenue")
plt.ylabel("Savings Objective")
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()


# In[56]:


# Create frequency tables for each reason column
reason_equity_table = df_finance['Reason_Equity'].value_counts().reset_index(name='Count')
reason_mutual_table = df_finance['Reason_Mutual'].value_counts().reset_index(name='Count')
reason_bonds_table = df_finance['Reason_Bonds'].value_counts().reset_index(name='Count')

# Print or visualize the tables (e.g., using pandas.DataFrame.plot)

# Conduct chi-square test (replace 'Investment_Avenues' with your actual column name)
from scipy.stats import chi2_contingency

contingency_table = pd.crosstab(df_finance['Investment_Avenues'], df_finance['Reason_Equity'])  # Replace with relevant reason column
chi2, pval, _, _ = chi2_contingency(contingency_table.values)

if pval < 0.05:
    print("There is a statistically significant association between investment avenues and reasons for choosing equity.")
else:
    print("There is no statistically significant association found.")


# In[60]:


import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency


# Create frequency tables for each reason column
reason_equity_table = df_finance['Reason_Equity'].value_counts().reset_index(name='Count')
reason_mutual_table = df_finance['Reason_Mutual'].value_counts().reset_index(name='Count')
reason_bonds_table = df_finance['Reason_Bonds'].value_counts().reset_index(name='Count')

# Plot frequency tables (bar charts) using the actual reason column names
reason_equity_table.plot.bar(x='Reason_Equity', y='Count', title="Reasons for Investing in Equity Market")
reason_mutual_table.plot.bar(x='Reason_Mutual', y='Count', title="Reasons for Investing in Mutual Funds")
reason_bonds_table.plot.bar(x='Reason_Bonds', y='Count', title="Reasons for Investing in Government Bonds")
plt.xticks(rotation=45)  # Rotate x-axis labels for readability
plt.tight_layout()  # Adjust spacing between plots
plt.show()

# Conduct chi-square test (replace 'Investment_Avenues' with your actual column name)
contingency_table = pd.crosstab(df_finance['Investment_Avenues'], df_finance['Reason_Equity'])  # Replace with relevant reason column
chi2, pval, _, _ = chi2_contingency(contingency_table.values)

# Interpret chi-square test result for stakeholders
if pval < 0.05:
    print("**Statistically Significant Association Found**")
    print("There is a statistically significant association between the investment avenues chosen and the reasons for choosing equity. This suggests that people's motivations for investing in equity may differ depending on the other investment options they consider.")
else:
    print("**No Statistically Significant Association Found**")
    print("Based on the data, there is no statistically significant association between investment avenues and reasons for choosing equity. This may indicate that similar reasons influence investment decisions across different avenues, or that our sample size might be insufficient to detect a clear association.")


# In[65]:


# Extract the column containing sources of information
information_sources = df_data['Your sources of information for investments is ']

# Check the frequency of each information source
source_counts = information_sources.value_counts()

# Display the frequency table of information sources
print("Frequency of Information Sources for Investments:")
print(source_counts)


# In[66]:


# Plotting the frequency of information sources
plt.figure(figsize=(10, 6))
source_counts.plot(kind='bar', color='skyblue')
plt.title('Primary Sources of Investment Information')
plt.xlabel('Information Sources')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[ ]:




