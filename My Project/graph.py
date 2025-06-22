import matplotlib.pyplot as plt
import numpy as np

# Sample data
# Sample data
categories = ['DenseNet', 'Inception', 'InceptionResnet', 'Resnet','VGG16','VGG16(Random_forest)']
values1 = [91, 80, 64, 41, 55, 79]
values2 = [64, 62, 58, 29, 44, 50]  # Second set of values

# Number of categories
n = len(categories)

# The x locations for the groups (introducing gaps)
x = np.arange(n) * 1.5  # Increasing the distance between groups by multiplying by 1.5

# The width of the bars
width = 0.35

# Create the plot
fig, ax = plt.subplots(figsize=(12, 6))  # Adjust figure size for better spacing

# Plot bars for the first set of values
bars1 = ax.bar(x - width/2, values1, width, label='maturity', color='blue')

# Plot bars for the second set of values
bars2 = ax.bar(x + width/2, values2, width, label='Quality', color='green')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Models')
ax.set_ylabel('Accuracy')
ax.set_title('Grouped Bar Chart with Rotated Labels and Increased Distance Between Groups')
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=45, ha='right')  # Rotate labels and align them to the right
ax.legend()

# Annotate each bar with the value
def annotate_bars(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate('{}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

annotate_bars(bars1)
annotate_bars(bars2)

# Save the plot as an image file
plt.savefig('grouped_bar_graph_with_rotated_labels.png')

# Show the plot
plt.show()





