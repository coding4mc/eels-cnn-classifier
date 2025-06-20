from typing import List, Callable, Dict
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, callback_context

from .plot_1d import Plot1D


class PlotViewer1D:
    def __init__(self, plots: List[Plot1D]) -> None:
        self.plots = plots
        self.app = Dash(__name__)
        
        # Create the layout
        self.app.layout = html.Div([
            html.H1("Plot Viewer", style={'textAlign': 'center'}),
            
            # Plot display
            dcc.Graph(id='plot-display', style={'height': '60vh'}),
            
            # Navigation controls
            html.Div([
                html.Button("Previous", id='prev-button', n_clicks=0),
                html.Button("Next", id='next-button', n_clicks=0),
                html.Div(id='plot-index', style={'display': 'none'}, children='0')
            ], style={'textAlign': 'center', 'padding': '20px'}),
            
        ], style={'padding': '20px'})
        
        # Callback for updating the plot
        @self.app.callback(
            Output('plot-display', 'figure'),
            Output('plot-index', 'children'),
            Input('prev-button', 'n_clicks'),
            Input('next-button', 'n_clicks'),
            Input('plot-index', 'children')
        )
        def update_plot(prev_clicks: int, next_clicks: int, current_index: str) -> tuple:
            ctx = callback_context
            
            # Get current index
            index = int(current_index)
            
            # Determine which button was clicked
            if ctx.triggered:
                button_id = ctx.triggered[0]['prop_id'].split('.')[0]
                if button_id == 'prev-button':
                    index = (index - 1) % len(self.plots)
                elif button_id == 'next-button':
                    index = (index + 1) % len(self.plots)
            
            # Return current plot and index
            plot = self.plots[index]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=plot.x, y=plot.y, mode='lines', name=plot.title))
            fig.update_layout(title=plot.title, xaxis_title='x', yaxis_title='y')
            return fig, str(index)
    
    def run(self, debug: bool = False, port: int = 8050) -> None:
        self.app.run(debug=debug, port=port)
