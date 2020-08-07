import os
from glob import glob
import base64
from io import BytesIO
import time

#import flask
from flask import send_file, make_response 
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from PIL import Image

from model import detect_scores_bboxes_classes, filter_boxes, detr
from model import CLASSES, DEVICE 
print(f"ObjectDetect loaded, using DEVICE={DEVICE}")

# ----------
# Helper functions 
def getImageFileNames(dirPath):
    if not os.path.isdir(dirPath):
        return go.Figure().update_layout(title='Incorrect Directory Path!')
    
    for filetype in ('*.png', '*.jpg'):
        fnames = sorted(glob(os.path.join(dirPath,filetype)))
        if len(fnames) > 0: break

    if not fnames:
        go.Figure().update_layout(title='No files found!')
        return None
    else:
        return fnames


# ----------
# Dash component wrappers
def Row(children=None, **kwargs):
    return html.Div(children, className="row", **kwargs)


def Column(children=None, width=1, **kwargs):
    nb_map = {
        1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six',
        7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten', 11: 'eleven', 12: 'twelve'}

    return html.Div(children, className=f"{nb_map[width]} columns", **kwargs)

# ----------
# plotly.py helper functions

def pil_to_b64(im, enc="png"):
    io_buf = BytesIO()
    im.save(io_buf, format=enc)
    encoded = base64.b64encode(io_buf.getvalue()).decode("utf-8")
    return f"data:img/{enc};base64, " + encoded


def pil_to_fig(im, showlegend=False, title=None):
    img_width, img_height = im.size
    fig = go.Figure()
    # This trace is added to help the autoresize logic work.
    fig.add_trace(go.Scatter(
        x=[img_width * 0.05, img_width * 0.95],
        y=[img_height * 0.95, img_height * 0.05],
        showlegend=False, mode="markers", marker_opacity=0, 
        hoverinfo="none", legendgroup='Image'))

    fig.add_layout_image(dict(
        source=pil_to_b64(im), sizing="stretch", opacity=1, layer="below",
        x=0, y=0, xref="x", yref="y", sizex=img_width, sizey=img_height,))

    # Adapt axes to the right width and height, lock aspect ratio
    fig.update_xaxes(
        showgrid=False, visible=False, constrain="domain", range=[0, img_width])
    
    fig.update_yaxes(
        showgrid=False, visible=False,
        scaleanchor="x", scaleratio=1,
        range=[img_height, 0])
    
    fig.update_layout(title=title, showlegend=showlegend)

    return fig


def add_bbox(fig, x0, y0, x1, y1, 
             showlegend=True, name=None, color=None, 
             opacity=0.5, group=None, text=None):
    fig.add_trace(go.Scatter(
        x=[x0, x1, x1, x0, x0],
        y=[y0, y0, y1, y1, y0],
        mode="lines",
        fill="toself",
        opacity=opacity,
        marker_color=color,
        hoveron="fills",
        name=name,
        hoverlabel_namelength=0,
        text=text,
        legendgroup=group,
        showlegend=showlegend,
    ))

# colors for visualization
COLORS = ['#fe938c','#86e7b8','#f9ebe0','#208aae','#fe4a49', 
          '#291711', '#5f4b66', '#b98b82', '#87f5fb', '#63326e'] * 50

# Start Dash
app = dash.Dash(__name__)
server = app.server  # Expose the server variable for deployments

# ----------
# layout
app.layout = html.Div(className='container', children=[
    Row(html.H1("Video Object Removal App")),

    Row(html.P("Input Directory Path:")),
    Row([
        Column(width=6, children=[
            dcc.Input(id='input-dirpath', style={'width': '100%'}, placeholder='Insert dirpath...'),
        ]),
        Column(html.Button("Run Single", id='button-single', n_clicks=0), width=2),
        Column(html.Button("Run Sequence", id='button-sequence', n_clicks=0), width=2),
        Column(html.Button("Run Inpaint", id='button-inpaint', n_clicks=0), width=2)
    ]),

    html.Hr(),

    Row([ 
         Column(width=2, children=[ html.P('Frame range:')]),
         Column(width=10, children=[ 
                dcc.Input(id='input-framenmin',type='number',value=0),
                dcc.RangeSlider(
                    id='slider-framenums', min=0, max=100, step=1, value=[0,100],
                    marks={0: '0', 100: '100'}, allowCross=False),
                dcc.Input(id='input-framenmax',type='number',value=100)
            ], style={"display": "grid", "grid-template-columns": "10% 70% 10%"}),
    ],style={'width':"100%"}),
    
    html.Hr(),

    Row([
        Column(width=7, children=[
            html.P('Non-maximum suppression (IoU):'),
            Row([
                Column(width=3, children=dcc.Checklist(
                    id='checklist-nms', 
                    options=[{'label': 'Enabled', 'value': 'enabled'}],
                    value=[])),

                Column(width=9, children=dcc.Slider(
                    id='slider-iou', min=0, max=1, step=0.05, value=0.5, 
                    marks={0: '0', 1: '1'})),
            ])
        ]),
        Column(width=5, children=[
            html.P('Confidence Threshold:'),
            dcc.Slider(
                id='slider-confidence', min=0, max=1, step=0.05, value=0.7, 
                marks={0: '0', 1: '1'})
        ])
    ]),

    Row(dcc.Graph(id='model-output', style={"height": "70vh"})),

    html.Hr(),

    Row([
        Column(width=2, children=[html.P("Current progress")]),
        Column(width=12, children=[html.Progress(id='progress-sequence',max=100,value=23)])
    ]),

    html.Video(id='sequence-output',src='/static/result.mp4',controls=True,style={"height": "70vh"})
])


# ----------
# callbacks 

# update_framenum_minmax()
# purpose:  to update min/max boxes of the slider 
@app.callback(
    [Output('input-framenmin','value'),
     Output('input-framenmax','value')],
    [Input('slider-framenums','value')]
)
def update_framenum_minmax(framenumrange):
    return framenumrange[0], framenumrange[1]


@app.callback(
    [Output('slider-framenums','max'),
     Output('slider-framenums','marks'),
     Output('slider-framenums','value'),
     Output('input-dirpath','value')    ],
    [Input('button-single','n_clicks'),
     Input('button-sequence','n_clicks'),
     Input('button-inpaint', 'n_clicks')],
    [State('input-dirpath', 'value'),
     State('slider-framenums','max'),
     State('slider-framenums','marks'),
     State('slider-framenums','value')  ]
)
def update_dirpath(nc_single, nc_sequence, nc_inpaint, s_dirpath, s_fnmax, s_fnmarks, s_fnvalue):
    if s_dirpath is None:
        s_dirpath = "/home/appuser/data/Colomar/frames"  #temporary fix
    dirpath = s_dirpath
    fnames = getImageFileNames(s_dirpath)
    if fnames:
        fnmax = len(fnames)-1
        if fnmax != s_fnmax: 
            fnmarks = {0: '0', fnmax: f"{fnmax}"}
            fnvalue = [0, fnmax]
        else:
            fnmarks = s_fnmarks
            fnvalue = s_fnvalue
    else:
        fnmax = s_fnmax
        fnmarks = s_fnmarks
        fnvalue = s_fnvalue
        
    return fnmax, fnmarks, fnvalue, dirpath


# update_framenum_slider()
# purpose:  to update position of the slider 
#@app.callback(
#    [Output('slider-framenums','value')],
#    [Input('input-framenmin','value'),
#     Input('input-framenmax','value')]
#)
#def update_framenum_slider(framenummin,framnummax):
#    return [framenummin, framnummax] 


#@app.callback(
#    [Output('button-single', 'n_clicks'),
#     Output('slider-framenums','max'),
#     Output('slider-framenums','marks'),
#     Output('slider-framenums','value'),
#     Output('input-dirpath', 'value')],
#    [Input('button-sequence', 'n_clicks')],
#    [State('button-single', 'n_clicks')])
#def run_sequence(random_n_clicks, run_n_clicks):
#    #return run_n_clicks+1, RANDOM_URLS[random_n_clicks%len(RANDOM_URLS)]
#    dirpath = "/home/appuser/data/Colomar/frames" 
#    fnames = getImageFileNames(dirpath)
#    fnmax = len(fnames)-1
#    marks = {0: '0', fnmax: f"{fnmax}"}
#    return run_n_clicks+1, fnmax, marks, [0,fnmax], dirpath


@app.callback(
    [Output('model-output', 'figure'),
     Output('slider-iou', 'disabled')],
    [Input('button-single', 'n_clicks')],
    [State('input-dirpath', 'value'),
     State('slider-iou', 'value'),
     State('slider-framenums','value'),
     State('slider-confidence', 'value'),
     State('checklist-nms', 'value')],
)
def run_single(n_clicks, dirpath, iou, framerange, confidence, checklist):
    apply_nms = 'enabled' in checklist
    if dirpath is not None and os.path.isdir(dirpath):
        fnames = getImageFileNames(dirpath)
        imgfile = fnames[framerange[0]]
        im = Image.open(imgfile)
    else: 
        go.Figure().update_layout(title='Incorrect dirpath')
        im = Image.new('RGB',(640,480))
        fig = pil_to_fig(im, showlegend=True, title='No Image')
        return fig, not apply_nms

    tstart = time.time()
    scores, boxes, selClasses = detect_scores_bboxes_classes(imgfile, detr)
    tend = time.time()

    fig = pil_to_fig(im, showlegend=True, title=f'DETR Predictions ({tend-tstart:.2f}s)')
    existing_classes = set()

    for confidence,bbx,class_id in zip(scores,boxes,selClasses):
        x0, y0, x1, y1 = bbx 
        label = CLASSES[class_id]

        # only display legend when it's not in the existing classes
        showlegend = label not in existing_classes
        text = f"class={label}<br>confidence={confidence:.3f}"

        add_bbox(
            fig, x0, y0, x1, y1,
            opacity=0.7, group=label, name=label, color=COLORS[class_id], 
            showlegend=showlegend, text=text,
        )

        existing_classes.add(label)

    return fig, not apply_nms


@app.callback(
    [Output('progress-sequence', 'value')],
    [Input('button-sequence', 'n_clicks')],
    [State('input-dirpath', 'value'),
     State('slider-framenums','value'),
     State('slider-confidence', 'value')]
)
def run_sequence(n_clicks, dirpath, framerange, confidence):

    if dirpath is not None and os.path.isdir(dirpath):
        fnames = getImageFileNames(dirpath)
    else: 
        return [0] 
    
    fmin, fmax = framerange
    fnames = fnames[fmin:fmax]

    # was this a repeat?
    if len(detr.imglist) != 0:
        if fnames == detr.selectFiles:
            return [0]
        else:
            detr.__init__()

    detr.selectFiles = fnames

    staticdir = os.path.join(os.getcwd(),"static")
    detr.load_images(filelist=fnames)
    detr.predict_sequence()
    detr.groupObjBBMaskSequence()
    detr.create_animationObject(framerange=framerange,
                                useMasks=True,
                                toHTML=False,
                                figsize=(20,15),
                                interval=30,
                                MPEGfile=os.path.join(staticdir,'result.mp4'))
    return [0] 



# simply forces refresh of video object
#@server.after_request
@server.route('/static/<path:path>')
def serve_video(inpath):
    return inpath

#@server.route('/static/<vid_name>')
#def serve_video(vid_name):
#    root_dir = os.path.join(os.getcwd(),'static')
#    vid_path = os.path.join(root_dir,vid_name)
#    resp = make_response(send_file(vid_path,'video/mp4'))
#    resp.headers['Content-Disposition'] = 'inline'
#    return resp 

#@server.after_request
#def add_header(response):
#    #response.cache_control.max_age = 1
#    if 'Cache-Control' not in response.headers:
#        response.headers['Cache-Control'] = 'no-store'
#    return response

#@app.callback(
#    [Output()]
#    [Input('download-button')]
#)
#def download_video(n_clicks):
#    root_dir = os.getcwd()
#    path='test.mov'
#    return flask.send_from_directory(os.path.join(root_dir,'static'),path)



# ---------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True,host='0.0.0.0')
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
