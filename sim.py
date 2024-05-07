import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Example state coordinates and initial conditions
coordinates = {
    'California': {'lat': 36.7783, 'lon': -119.4179},
    'Nevada': {'lat': 38.8026, 'lon': -116.4194},
    'Oregon': {'lat': 44.5720, 'lon': -122.0709},
    'Washington': {'lat': 47.7511, 'lon': -120.7401},
    'Arizona': {'lat': 34.0489, 'lon': -111.0937}
}

# Total populations for simplicity (in millions for easier calculations)
total_populations = {
    'California': 39.51,
    'Nevada': 3.08,
    'Oregon': 4.22,
    'Washington': 7.62,
    'Arizona': 7.28
}

# Initializing data for the epidemic model
data = {
    'infected': {state: np.random.randint(50, 100) for state in coordinates},
    'recovered': {state: np.random.randint(10, 50) for state in coordinates},
    'deceased': {state: np.random.randint(1, 10) for state in coordinates}
}

# Calculate initial healthy population
healthy = {state: total_populations[state] - (data['infected'][state] + data['recovered'][state] + data['deceased'][state]) for state in coordinates}

# Create subplot structure specifying subplot types
fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=("Spread of Disease", "Proportion of Population Status"),
    specs=[[{"type": "scattergeo"}], [{"type": "xy"}]],  # Use a Cartesian plot for the area chart
    row_heights=[0.7, 0.3]
)

# Add initial data for map and bar chart
for state, coord in coordinates.items():
    fig.add_trace(go.Scattergeo(
        lon=[coord['lon']],
        lat=[coord['lat']],
        text=f"{state}: {data['infected'][state]} infected",
        marker=dict(size=data['infected'][state]/10, color='red'),
        showlegend=False
    ), row=1, col=1)

# Timeseries x values
x_values = list(range(10))  # simulate 10 days

# Initialize the figure with data for the area chart
fig.add_trace(go.Scatter(
    x=x_values,
    y=[healthy[state] for state in coordinates],
    stackgroup='one',  # define stack group
    name='Healthy'
), row=2, col=1)
fig.add_trace(go.Scatter(
    x=x_values,
    y=[data['infected'][state] for state in coordinates],
    stackgroup='one',
    name='Infected',
    fill='tonexty'
), row=2, col=1)
fig.add_trace(go.Scatter(
    x=x_values,
    y=[data['recovered'][state] for state in coordinates],
    stackgroup='one',
    name='Recovered',
    fill='tonexty'
), row=2, col=1)
fig.add_trace(go.Scatter(
    x=x_values,
    y=[data['deceased'][state] for state in coordinates],
    stackgroup='one',
    name='Deceased',
    fill='tonexty'
), row=2, col=1)

# Create frames for the animation
frames = []
steps = 10  # Number of simulation steps

for step in range(steps):
    frame_data = []
    for state, coord in coordinates.items():
        # Random infection dynamics
        new_infected = np.random.randint(0, 50)
        new_recovered = np.random.randint(0, 20)
        new_deceased = np.random.randint(0, 5)

        # Update data
        data['infected'][state] += new_infected
        data['recovered'][state] += new_recovered
        data['deceased'][state] += new_deceased
        healthy[state] = max(total_populations[state] - (data['infected'][state] + data['recovered'][state] + data['deceased'][state]), 0)

        # Update data for this frame
        frame_data.append(go.Scattergeo(
            lon=[coord['lon']],
            lat=[coord['lat']],
            text=f"{state}: {data['infected'][state]} infected",
            marker=dict(size=data['infected'][state]/10, color='red'),
            showlegend=False
        ))

    frame_data.extend([
        go.Scatter(x=x_values, y=[healthy[state] for state in coordinates], stackgroup='one'),
        go.Scatter(x=x_values, y=[data['infected'][state] for state in coordinates], stackgroup='one', fill='tonexty'),
        go.Scatter(x=x_values, y=[data['recovered'][state] for state in coordinates], stackgroup='one', fill='tonexty'),
        go.Scatter(x=x_values, y=[data['deceased'][state] for state in coordinates], stackgroup='one', fill='tonexty')
    ])

    frames.append(go.Frame(data=frame_data, name=str(step)))

# Set up frames
fig.frames = frames

# Add animation controls
fig.update_layout(
    updatemenus=[{
        "buttons": [
            {
                "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True, "transition": {"duration": 300}}],
                "label": "Play",
                "method": "animate"
            }
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 87},
        "showactive": False,
        "type": "buttons",
        "x": 0.1,
        "xanchor": "right",
        "y": 0,
        "yanchor": "top"
    }],
    geo=dict(
        scope='usa',
        landcolor='rgb(217, 217, 217)'
    )
)

fig.show()
