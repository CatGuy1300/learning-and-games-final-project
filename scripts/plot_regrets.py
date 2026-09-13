import matplotlib.pyplot as plt

# Data extracted from "image_d3ac55.png"
number_of_actions = [2, 3, 6, 10]
worst_case_regret = [11.59, 27.08, 86.87, 185.03]

# Plot the points and connect them with a line
plt.plot(number_of_actions, worst_case_regret, marker='o', linestyle='-', color='blue')

# Add labels and a title to match the table
plt.xlabel('Number of Actions')
plt.ylabel('Worst Case Regret')
plt.title('Worst-case regret based on the number of actions')

# Add a grid for readability
plt.grid(True)

# Display the plot
plt.show()