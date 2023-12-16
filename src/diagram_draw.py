import pandas as pd
import matplotlib.pyplot as plt

# Create a DataFrame from the provided data
data_independable = {
    'Motion Category': ['uni_lin', 'uni_lin', 'uni_lin',
                        'uni_cir', 'uni_cir', 'uni_cir',
                        'acc_lin', 'acc_lin', 'acc_lin',
                        'acc_cir', 'acc_cir', 'acc_cir'],
    'Number of Fault Module': [0, 1, 2,
                               0, 1, 2,
                               0, 1, 2,
                               0, 1, 2],
    'Accuracy': [92.40506329113924, 91.77, 87.34,
                 93.78238341968912, 90.67, 85.98,
                 89.8876404494382, 87.64, 81.89,
                 79.6116504854369, 75.73, 69.90]
}

data_dependable = {
    'Motion Category': ['uni_lin', 'uni_lin', 'uni_lin',
                        'uni_cir', 'uni_cir',  'uni_cir',
                        'acc_lin', 'acc_lin',  'acc_lin',
                        'acc_cir', 'acc_cir' , 'acc_cir'],
    'Number of Fault Module': [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
    'Accuracy': [92.40506329113924, 91.77215189873418, 91.13924050632912,
                 93.78238341968912, 92.2279792746114, 92.74611398963731,
                 89.8876404494382, 89.8876404494382, 89.8876404494382,
                 79.6116504854369, 79.6116504854369, 79.6116504854369]
}

data = data_dependable
# data = data_independable

df = pd.DataFrame(data)

# Pivot the DataFrame to get the correct format for a grouped bar chart
pivot_df = df.pivot(index='Motion Category', columns='Number of Fault Module', values='Accuracy')

# Plotting
pivot_df.plot(kind='bar', figsize=(10, 7))

# Customizing the plot
# plt.title('Accuracy by Motion Category and Number of Fault Module')
plt.xlabel('Motion Category')
plt.ylabel('Accuracy (%)')
plt.xticks(rotation=0)
plt.legend(title='Number of Fault Module')
plt.ylim(0, 100)  # Set the limit for y-axis to 0-100 since these are percentages

# Display the plot
plt.tight_layout()
plt.show()
plt.savefig('errors_not_exceed_max_threshold_accuracy.png', dpi=500)
plt.clf()

data = data_independable

df = pd.DataFrame(data)

# Pivot the DataFrame to get the correct format for a grouped bar chart
pivot_df = df.pivot(index='Motion Category', columns='Number of Fault Module', values='Accuracy')

# Plotting
pivot_df.plot(kind='bar', figsize=(10, 7))

# Customizing the plot
# plt.title('Accuracy by Motion Category and Number of Fault Module')
plt.xlabel('Motion Category')
plt.ylabel('Accuracy (%)')
plt.xticks(rotation=0)
plt.legend(title='Number of Fault Module')
plt.ylim(0, 100)  # Set the limit for y-axis to 0-100 since these are percentages

# Display the plot
plt.tight_layout()
plt.show()
plt.savefig('errors_exceed_max_threshold_accuracy.png', dpi=500)
plt.clf()