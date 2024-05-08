import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('H:/downloads/US.csv')

# Convert the 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Calculate the 7-day moving averages
df['confirmed_7d_avg'] = df['new_confirmed'].rolling(window=7).mean()
df['deceased_7d_avg'] = df['new_deceased'].rolling(window=7).mean()
df['tested_7d_avg'] = df['new_tested'].rolling(window=7).mean()
actual_data = df['confirmed_7d_avg']#.iloc[620:850].values
dates = df['date']#.iloc[620:850].values

# actual_data = df['confirmed_7d_avg'].iloc[0:150].values
# dates = df['date'].iloc[0:150].values

# Plotting
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))
axes.plot(dates, actual_data, color='orange', label='7-Day Average')
axes.set_title('7-Day Average of New Confirmed Cases')
axes.set_xlabel('Date')
axes.set_ylabel('Number of Cases')
axes.legend()

# # Plot new_confirmed cases and its 7-day average
# #axes[0].plot(df['date'], df['new_confirmed'], marker='o', color='b', label='Daily Confirmed')
# axes[0].plot(df['date'], df['confirmed_7d_avg'], color='orange', label='7-Day Average')
# axes[0].set_title('New Confirmed Cases and 7-Day Average')
# axes[0].set_xlabel('Date')
# axes[0].set_ylabel('Number of Cases')
# axes[0].legend()

# # Plot new_deceased cases and its 7-day average
# #axes[1].plot(df['date'], df['new_deceased'], marker='o', color='r', label='Daily Deceased')
# axes[1].plot(df['date'], df['deceased_7d_avg'], color='purple', label='7-Day Average')
# axes[1].set_title('New Deceased Cases and 7-Day Average')
# axes[1].set_xlabel('Date')
# axes[1].set_ylabel('Number of Cases')
# axes[1].legend()

# # Plot new_tested cases and its 7-day average
# #axes[2].plot(df['date'], df['new_tested'], marker='o', color='g', label='Daily Tested')
# axes[2].plot(df['date'], df['tested_7d_avg'], color='cyan', label='7-Day Average')
# axes[2].set_title('New Tested Cases and 7-Day Average')
# axes[2].set_xlabel('Date')
# axes[2].set_ylabel('Number of Cases')
# axes[2].legend()

# Improve layout and display the plots
plt.tight_layout()
plt.show()