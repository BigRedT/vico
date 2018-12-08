import plotly
import plotly.graph_objs as go

def create_sim_heatmap(sim_mat,xlabels,ylabels,filename):
    trace = go.Heatmap(
        z=sim_mat[::-1],
        x=xlabels,
        y=ylabels[::-1],
        showscale=True)
    data = [trace]

    layout = go.Layout(
        yaxis=dict(
            tickfont=dict(size=10),
        ),
        xaxis=dict(
            tickangle=-45,
            side='top',
            tickfont=dict(size=10),
        ),
        height=800,
        width=800,
        autosize=False,
        showlegend=False,
        margin=go.layout.Margin(
            l=150,
            r=100,
            b=100,
            t=100,
        ),
    )

    plotly.offline.plot(
        {'data': [trace], 'layout': layout},
        filename=filename,
        auto_open=False)