from typing import List, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.graph_objects import ColorBar
from dash import Dash, html, dcc, Input, Output, State, ctx
from dash.exceptions import PreventUpdate
import numpy as np

from .types import Coordinate, GraphClickData
from .plot_heatmap import PlotHeatmap
from .plot_3d import Plot3D
from .plot_viewer_page_data import PlotViewerPageData


class PlotViewer3D:

    _LEFT_BUTTON_ID = "left-button"
    _RIGHT_BUTTON_ID = "right-button"
    _UP_BUTTON_ID = "up-button"
    _DOWN_BUTTON_ID = "down-button"
    _LEFT_PAGE_BUTTON_ID = "left-page-button"
    _RIGHT_PAGE_BUTTON_ID = "right-page-button"
    _PAGE_INDEX_TEXT_ID = "page-index-text"
    _PAGE_INDEX_STORE_ID = "page-index-store"


    def __init__(
            self,
            page_data_list: List[PlotViewerPageData],
            title: Optional[str] = None
    ):
        self.page_data_list = page_data_list
        
        self.app = Dash(__name__)
        
        # Create the layout
        self.app.layout = html.Div(id="main", children=[
            html.H1(title, style={'textAlign': 'center'}) if title else None,

            html.Div(
                children=[
                    html.Button("Previous", id=self._LEFT_PAGE_BUTTON_ID,
                        style={
                            'textAlign': 'center',
                            'margin': '10px'
                        }
                    ),
                    html.Span("Page index: 0", id=self._PAGE_INDEX_TEXT_ID,
                        style={
                            'textAlign': 'center',
                            'margin': '10px'
                        }
                    ),
                    html.Button("Next", id=self._RIGHT_PAGE_BUTTON_ID,
                        style={
                            'textAlign': 'center',
                            'margin': '10px'
                        }
                    ),
                ],
                style={'display': 'flex', 'justify-content': 'center'}
            ),

            # Main plot container
            dcc.Graph(id='main-plot', style={'height': '80vh'}),
            
            # Store clicked coordinates
            dcc.Store(id='clicked-coordinates', data={'x': 0, 'y': 0}),

            # Store page
            dcc.Store(id=self._PAGE_INDEX_STORE_ID, data=0),

            # Buttons for keyboard events
            html.Button(self._LEFT_BUTTON_ID, id=self._LEFT_BUTTON_ID, hidden=True),
            html.Button(self._RIGHT_BUTTON_ID, id=self._RIGHT_BUTTON_ID, hidden=True),
            html.Button(self._UP_BUTTON_ID, id=self._UP_BUTTON_ID, hidden=True),
            html.Button(self._DOWN_BUTTON_ID, id=self._DOWN_BUTTON_ID, hidden=True),
        ])

        self.app.clientside_callback(
            f"""
                function(id) {{
                    window.addEventListener("keydown", function(event) {{
                        if (event.key == "ArrowLeft") {{
                            // Check MacOS command key
                            if (event.metaKey) {{
                                document.getElementById('{self._LEFT_PAGE_BUTTON_ID}').click();
                            }} else {{
                                document.getElementById('{self._LEFT_BUTTON_ID}').click();
                            }}
                            event.preventDefault();
                            return false;
                        }} else if (event.key == "ArrowRight") {{
                            // Check MacOS command key
                            if (event.metaKey) {{
                                document.getElementById('{self._RIGHT_PAGE_BUTTON_ID}').click();
                            }} else {{
                                document.getElementById('{self._RIGHT_BUTTON_ID}').click();
                            }}
                            event.preventDefault();
                            return false;
                        }} else if (event.key == "ArrowUp") {{
                            console.log(event.key);
                            document.getElementById('{self._UP_BUTTON_ID}').click();
                            event.preventDefault();
                            return false;
                        }} else if (event.key == "ArrowDown") {{
                            console.log(event.key);
                            document.getElementById('{self._DOWN_BUTTON_ID}').click();
                            event.preventDefault();
                            return false;
                        }}
                        return true;
                    }}, false);
                    return window.dash_clientside.no_update       
                }}
            """,
            Output("main", "id"),
            Input("main", "id")
        )

        @self.app.callback(
            Output(self._PAGE_INDEX_STORE_ID, 'data'),
            Output(self._PAGE_INDEX_TEXT_ID, 'children'),
            Output(self._LEFT_PAGE_BUTTON_ID, 'disabled'),
            Output(self._RIGHT_PAGE_BUTTON_ID, 'disabled'),
            Input(self._PAGE_INDEX_STORE_ID, 'data'),
            Input(self._LEFT_PAGE_BUTTON_ID, 'n_clicks'),
            Input(self._RIGHT_PAGE_BUTTON_ID, 'n_clicks'),
        )
        def _on_page_button_change(
            page_index: int,
            left_page_button,
            right_page_button
        ) -> Tuple[int, str, bool, bool]:            
            if ctx.triggered_id == self._LEFT_PAGE_BUTTON_ID:
                wanted_page_index = page_index - 1
            elif ctx.triggered_id == self._RIGHT_PAGE_BUTTON_ID:
                wanted_page_index = page_index + 1
            else:
                raise PreventUpdate("Change page buttons not pressed.")

            new_page_index = max(0, min(len(self.page_data_list) - 1, wanted_page_index))
            page_index_text = f"Page index: {new_page_index}"
            left_button_disabled = new_page_index == 0
            right_button_disabled = new_page_index == len(self.page_data_list) - 1
            return new_page_index, page_index_text, left_button_disabled, right_button_disabled

        
        # Callback for click events
        @self.app.callback(
            Output('main-plot', 'figure'),
            Output('clicked-coordinates', 'data'),
            Input('main-plot', 'clickData'),
            Input('clicked-coordinates', 'data'),
            Input(self._PAGE_INDEX_STORE_ID, 'data'),
            Input(self._LEFT_BUTTON_ID, 'n_clicks'),
            Input(self._RIGHT_BUTTON_ID, 'n_clicks'),
            Input(self._UP_BUTTON_ID, 'n_clicks'),
            Input(self._DOWN_BUTTON_ID, 'n_clicks'),
            Input(self._LEFT_PAGE_BUTTON_ID, 'n_clicks'),
            Input(self._RIGHT_PAGE_BUTTON_ID, 'n_clicks'),
        )
        def _update_plot(
            clickData: GraphClickData,
            stored_coords: Coordinate,
            page_index: int,
            left, right, up, down,
            left_page, right_page
        ) -> Tuple[go.Figure, Coordinate]:
            # Update page if page buttons are clicked
            page_just_changed = False
            if ctx.triggered_id == self._LEFT_PAGE_BUTTON_ID and page_index > 0:
                page_just_changed = True
            elif ctx.triggered_id == self._LEFT_PAGE_BUTTON_ID and page_index < len(self.page_data_list) - 1:
                page_just_changed = True
            
            # Update stored coords
            figure, stored_coords = _draw_page(
                clickData=clickData,
                stored_coords=stored_coords,
                page_index=page_index,
                page_just_changed=page_just_changed
            )
            return figure, stored_coords
            
        def _draw_page(
                clickData: GraphClickData,
                stored_coords: Coordinate,
                page_index: int,
                page_just_changed: bool
        ) -> Tuple[go.Figure, Coordinate]:
            x, y = stored_coords['x'], stored_coords['y']
            page_data = self.page_data_list[page_index]
            
            # Update coordinates if there's a left/right/up/down key press
            if page_just_changed:
                x, y = 0, 0
            elif ctx.triggered_id == self._LEFT_BUTTON_ID:
                x = max(0, x - 1)
            elif ctx.triggered_id == self._RIGHT_BUTTON_ID:
                x = min(page_data.width - 1, x + 1)
            elif ctx.triggered_id == self._UP_BUTTON_ID:
                y = min(page_data.height - 1, y + 1)
            elif ctx.triggered_id == self._DOWN_BUTTON_ID:
                y = max(0, y - 1)
            # Update coordinates if there's a click
            elif clickData is not None:
                x = int(clickData['points'][0]['x'])
                y = int(clickData['points'][0]['y'])
            
            stored_coords = {'x': x, 'y': y}

            plot_heatmaps = page_data.plot_heatmaps
            plot_3ds = page_data.plot_3ds

            row_count = max(len(plot_heatmaps), len(plot_3ds))

            # Create subplots
            subplot_titles = []
            for i in range(max(len(plot_heatmaps), len(plot_3ds))):
                heatmap_title = plot_heatmaps[i].title if i < len(plot_heatmaps) else ''
                plot_3d_title = plot_3ds[i].plots[y][x].title if i < len(plot_3ds) else ''

                subplot_titles.append(heatmap_title)
                subplot_titles.append('') # Colorbar on column 2
                subplot_titles.append(plot_3d_title or '') # No plots on column 3, so set empty title

            fig = make_subplots(
                rows=row_count,
                cols=3,
                specs=[
                    [
                        {} if i < len(plot_heatmaps) else None,
                        {},
                        {
                            "rowspan": max(1, min(3, len(plot_heatmaps) // len(plot_3ds)))
                        } if i < len(plot_3ds) else None
                    ]
                    for i in range(row_count)
                ],
                subplot_titles=subplot_titles,
                column_widths=[0.4, 0.1, 0.5],
            )
            
            # Plot heatmaps
            for index, heatmap in enumerate(plot_heatmaps):
                _draw_heatmap(
                    fig=fig,
                    heatmap=heatmap,
                    x=x,
                    y=y,
                    subplot_row=index + 1,
                    subplot_column=1,
                    heatmap_count=len(plot_heatmaps)
                )
            
            # Plot graph
            for index, plot_3d in enumerate(plot_3ds):
                _draw_graph(
                    fig=fig,
                    plot_3d=plot_3d,
                    x=x,
                    y=y,
                    subplot_row=index + 1,
                    subplot_column=3
                )
            
            # Update layout
            fig.update_layout(
                height=200 * row_count,
                showlegend=True,
                title=page_data.title
            )
            
            # Make heatmap aspect ratio equal
            for i in range(len(plot_heatmaps)):
                fig.update_xaxes(scaleanchor="y", scaleratio=1, row=i + 1, col=1)
            
            return fig, stored_coords
        
        
        def _draw_heatmap(
                fig: go.Figure,
                heatmap: PlotHeatmap,
                x: int,
                y: int,
                subplot_row: int,
                subplot_column: int,
                heatmap_count: int,

        ):
            # Add 2D matrix plot
            fig.add_trace(
                go.Heatmap(
                    z=heatmap.data,
                    # colorscale='plasma',
                    colorscale=[
                        [0, 'blue'], 
                        [0.5, 'white'],    
                        [1, 'orange']     # 3 will be orange
                    ],
                    showscale=True,
                    colorbar=dict(
                        x=0.37,  # Position colorbar between plots
                        y=0.9 - ((subplot_row - 1) * 1.13 / heatmap_count),  # Position based on row
                        len=1 / heatmap_count,  # Scale length based on number of heatmaps
                        title=heatmap.colourbar_title
                    ),
                    zmin=heatmap.colorbar_min,
                    zmax=heatmap.colorbar_max
                ),
                row=subplot_row, col=subplot_column
            )

            # Create a mask for the selected pixel
            mask = np.zeros_like(heatmap.data)
            mask[y, x] = 1

            # # Add selected pixel highlight
            # fig.add_trace(
            #     go.Heatmap(
            #         z=mask,
            #         showscale=False,
            #         colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(255,0,0,1)']],
            #         hoverinfo='skip'
            #     ),
            #     row=subplot_row, col=subplot_column
            # )

        def _draw_graph(
                fig: go.Figure,
                plot_3d: Plot3D,
                x: int,
                y: int,
                subplot_row: int,
                subplot_column: int
        ):
            # Add 1D graph for selected pixel
            plot_1d = plot_3d.plots[y][x]

            for curve in plot_1d.curves:
                color = curve.color
                fig.add_trace(
                    go.Scatter(
                        x=curve.x,
                        y=curve.y,
                        mode='lines',
                        name=curve.label,
                        line_color=f'rgba({color.r},{color.g},{color.b},1)' if color else None,
                    ),
                    row=subplot_row, col=subplot_column
                )
                if curve.std is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=curve.x,
                            y=curve.y + curve.std,
                            fill=None,
                            mode='lines',
                            line_color=f'rgba({color.r},{color.g},{color.b},0)' if color else None,
                            showlegend=False,
                            hoverinfo='skip'
                        ),
                        row=subplot_row, col=subplot_column
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=curve.x,
                            y=curve.y - curve.std,
                            fill='tonexty',  # fill area between traces
                            mode='lines',
                            line_color=f'rgba({color.r},{color.g},{color.b},0)' if color else None,
                            fillcolor=f'rgba({color.r},{color.g},{color.b},0.2)' if color else None,
                            showlegend=False,
                            hoverinfo='skip'
                        ),
                        row=subplot_row, col=subplot_column
                    )
                
    
    def run(self, debug: bool = False, port: int = 8050) -> None:
        self.app.run(debug=debug, port=port)
