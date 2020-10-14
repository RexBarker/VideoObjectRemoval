import os
import sys
import base64
from glob import glob
from shutil import rmtree
from io import BytesIO
import time
from datetime import datetime

#import flask
#from flask import send_file, make_response 
#from flask import send_from_directory
from flask_caching import Cache 
import dash
import dash_player
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from PIL import Image

from model import detect_scores_bboxes_classes, \
                  detr, createNullVideo
from model import CLASSES, DEVICE 
from model import inpaint, testContainerWrite, performInpainting

libpath = "/home/appuser/scripts/" # to keep the dev repo in place, w/o linking
sys.path.insert(1,libpath)
import ObjectDetection.imutils as imu


basedir = os.getcwd() 
uploaddir = os.path.join(basedir,"upload")
print(f"ObjectDetect loaded, using DEVICE={DEVICE}")
print(f"Basedirectory: {basedir}")

# cleaup old uploads
if os.path.exists(uploaddir):
    rmtree(uploaddir)    

os.mkdir(uploaddir)

# cleanup old runs
if os.path.exists(os.path.join(basedir,"static")):
    for f in [*glob(os.path.join(basedir,'static/sequence_*.mp4')),
              *glob(os.path.join(basedir,'static/inpaint_*.mp4'))]:
        os.remove(f)
else:
    os.mkdir(os.path.join(basedir,"static"))

# remove and create dummy video for place holder
tempfile = os.path.join(basedir,'static/result.mp4')
if not os.path.exists(tempfile):
    createNullVideo(tempfile)


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

# upload files
def save_file(name, content):
    """Decode and store a file uploaded with Plotly Dash."""
    filepath = os.path.join(uploaddir, name)
    data = content.encode("utf8").split(b";base64,")[1]
    with open(filepath, "wb") as fp:
        fp.write(base64.decodebytes(data))
    
    return filepath

def vfile_to_frames(fname):
    assert os.path.exists(fname), "Could not determine path to video file"
    dirPath = os.path.join(uploaddir,fname.split(".")[-2])
    nfiles = imu.videofileToFramesDirectory(videofile=fname,dirPath=dirPath,
                                            padlength=5, imgtype='png', cleanDirectory=True)
    return dirPath 

# PIL images
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

# graph 
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

def get_index_by_bbox(objGroupingDict,class_name,bbox_in):
    # gets the order of the object within the groupingSequence 
    # by bounding box overlap
    maxIoU = 0.0
    maxIdx = 0
    for instIdx, objInst in enumerate(objGroupingDict[class_name]):
        # only search for object found in seq_i= 0
        # there must exist objects at seq_i=0 since we predicted them on first frame
        bbx,_,seq_i = objInst[0] 

        if seq_i != 0: continue
        IoU = imu.bboxIoU(bbx, bbox_in)
        if IoU > maxIoU:
            maxIoU = IoU
            maxIdx = instIdx 
    
    return maxIdx


def get_selected_objects(figure,objGroupingDict):
    # recovers the selected items from the figure
    # infer the object location by the bbox IoU (more stable than order)
    # returns a dict of selected indexed items

    obsObjects = {}
    for d in figure['data']:
        objInst = d.get('legendgroup')
        wasVisible = False if d.get('visible') and d['visible'] == 'legendonly' else True
        if objInst is not None and ":" in objInst and wasVisible:

            classname = objInst.split(":")[0]
            x = d['x']
            y = d['y']
            bbox = [x[0], y[0], x[2], y[2]]
            inst = get_index_by_bbox(objGroupingDict,classname,bbox)
            if obsObjects.get(classname,None) is None:
                obsObjects[classname] = [int(inst)]
            else:
                obsObjects[classname].append(int(inst))
    
    return obsObjects


# colors for visualization
COLORS = ['#fe938c','#86e7b8','#f9ebe0','#208aae','#fe4a49', 
          '#291711', '#5f4b66', '#b98b82', '#87f5fb', '#63326e'] * 50

# Start Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # Expose the server variable for deployments
cache = Cache()
CACHE_CONFIG = {
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': os.path.join(basedir,'_cache')
} 
cache.init_app(server,config=CACHE_CONFIG)

# ----------
# layout
app.layout = html.Div(className='container', children=[
    Row(html.H1("Video Object Removal App")),

    # Input
    Row(html.P("Input Directory Path:")),
    Row([
        Column(width=6, children=[
            dcc.Upload(
                id="upload-file",
                children=html.Div(
                    ["Drag and drop or click to select a file to upload."]
                ),
                style={
                    "width": "100%",
                    "height": "60px",
                    "lineHeight": "60px",
                    "borderWidth": "1px",
                    "borderStyle": "dashed",
                    "borderRadius": "5px",
                    "textAlign": "center",
                    "margin": "10px",
                },
                multiple=False,
            )
        ]),
    ]),
 
    Row([
        Column(width=6, children=[
            html.H3('Upload file',id='input-dirpath-display', style={'width': '100%'})
        ]),
       Column(width=2,children=[
            html.Button("Run Single", id='button-single', n_clicks=0)
        ]),
    ]),

    html.Hr(),

    # frame number selection
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
        Column(width=5, children=[
            html.P('Confidence Threshold:'),
            dcc.Slider(
                id='slider-confidence', min=0, max=1, step=0.05, value=0.7, 
                marks={0: '0', 1: '1'})
        ]),
        Column(width=7, children=[
            html.P('Object selection:'),
            Row([
                Column(width=3, children=dcc.Checklist(
                    id='cb-person', 
                    options=[
                             {'label': ' person', 'value': 'person'},
                             {'label': ' handbag', 'value': 'handbag'},
                             {'label': ' backpack', 'value': 'backpack'},
                             {'label': ' suitcase', 'value': 'suitcase'},
                            ],
                    value = ['person', 'handbag','backpack','suitcase'])
                ),
                Column(width=3, children=dcc.Checklist(
                    id='cb-vehicle', 
                    options=[
                             {'label': ' car', 'value': 'car'},
                             {'label': ' truck', 'value': 'truck'},
                             {'label': ' bus', 'value': 'bus'},
                            ],
                    value = ['car', 'truck'])
                ),
                Column(width=3, children=dcc.Checklist(
                    id='cb-environment', 
                    options=[
                             {'label': ' traffic light', 'value': 'traffic light'},
                             {'label': ' stop sign', 'value': 'stop sign'},
                             {'label': ' bench', 'value': 'bench'},
                            ],
                    value = [])
                )
            ])
        ]),
    ]),

    # prediction output graph
    Row(dcc.Graph(id='model-output', style={"height": "70vh"})),

    # processing options
    html.Hr(),
    Row([
        Column(width=7, children=[
            html.P('Processing options:'),
            Row([
                Column(width=3, children=dcc.Checklist(
                    id='cb-options', 
                    options=[
                             {'label': ' use BBmask', 'value': 'useBBmasks'},
                             {'label': ' fill sequence', 'value': 'fillSequence'},
                            ],
                    value=['fillSequence'])
                ),
                Column(width=3, children= [
                    html.P('dilation half-width (no dilation=0):'),
                    dcc.Input(
                        id='input-dilationhwidth',
                        type='number',
                        value=0)
                ]),
                Column(width=3, children=[
                    html.P('minimum seq. length'),
                    dcc.Input(
                        id='input-minseqlength',
                        type='number',
                        value=0)
                ])
            ])
        ]),
    ]),
 
    # Sequence Video
    html.Hr(),
    Row([
        Column(width=2,children=[
            html.Button("Run Sequence", id='button-sequence', n_clicks=0),
            dcc.Loading(id='loading-sequence-bobble',
                        type='circle',
                        children=html.Div(id='loading-sequence'))
        ]),
        Column(width=2,children=[]), # place holder
        Row([
            Column(width=4, children=[
                html.P("Start Frame:"),
                html.Label("0",id='sequence-startframe')
            ]),
            Column(width=4, children=[]),
            Column(width=4, children=[
                html.P("End Frame:"),
                html.Label("0",id='sequence-endframe')
            ])
        ]),
    ]),
    html.P("Sequence Output"),
    html.Div([ dash_player.DashPlayer(
        id='sequence-output',
        url='static/result.mp4',
        controls=True,
        style={"height": "70vh"}) ]),

    # Inpainting Video
    html.Hr(),
    Row([
        Column(width=2,children=[
            html.Button("Run Inpaint", id='button-inpaint', n_clicks=0),
            dcc.Loading(id='loading-inpaint-bobble',
                        type='circle',
                        children=html.Div(id='loading-inpaint'))
        ]),
        Column(width=2,children=[]), # place holder
        Row([
            Column(width=4, children=[
                html.P("Start Frame:"),
                html.Label("0",id='inpaint-startframe')
            ]),
            Column(width=4, children=[]),
            Column(width=4, children=[
                html.P("End Frame:"),
                html.Label("0",id='inpaint-endframe')
            ])
        ]),
    ]),
    html.P("Inpainting Output"),
    html.Div([ dash_player.DashPlayer(
        id='inpaint-output',
        url='static/result.mp4',
        controls=True,
        style={"height": "70vh"}) ]),

    # hidden signal value
    html.Div(id='signal-sequence', style={'display': 'none'}),
    html.Div(id='signal-inpaint', style={'display': 'none'}),
    html.Div(id='input-dirpath', style={'display': 'none'}),

])


# ----------
# callbacks 

# upload file
@app.callback(
    Output("input-dirpath", "children"),
    [Input("upload-file", "filename"), Input("upload-file", "contents")],
)
def update_output(uploaded_filename, uploaded_file_content):
    # Save uploaded files and regenerate the file list.

    if uploaded_filename is not None and uploaded_file_content is not None and \
       uploaded_filename.split(".")[-1] in ('mp4', 'mov', 'avi') :
        dirPath = vfile_to_frames(save_file(uploaded_filename, uploaded_file_content))
        return dirPath
    else:
        return "(none)"

# update_framenum_minmax()
# purpose:  to update min/max boxes of the slider 
@app.callback(
    [Output('input-framenmin','value'),
     Output('input-framenmax','value'),
     Output('sequence-startframe','children'),
     Output('sequence-endframe','children'),
     Output('inpaint-startframe','children'),
     Output('inpaint-endframe','children')
     ],
    [Input('slider-framenums','value')]
)
def update_framenum_minmax(framenumrange):
    return framenumrange[0], framenumrange[1], \
           str(framenumrange[0]), str(framenumrange[1]), \
           str(framenumrange[0]), str(framenumrange[1])


@app.callback(
    [Output('slider-framenums','max'),
     Output('slider-framenums','marks'),
     Output('slider-framenums','value'),
     Output('input-dirpath-display','children')],
    [Input('button-single','n_clicks'),
     Input('button-sequence','n_clicks'),
     Input('button-inpaint', 'n_clicks'),
     Input('input-dirpath', 'children')],
    [State('upload-file', 'filename'),
     State('slider-framenums','max'),
     State('slider-framenums','marks'),
     State('slider-framenums','value')  ]
)
def update_dirpath(nc_single, nc_sequence, nc_inpaint, s_dirpath, 
                   vfilename, s_fnmax, s_fnmarks, s_fnvalue):
    if s_dirpath is None or s_dirpath == "(none)":
        #s_dirpath = "/home/appuser/data/Colomar/frames"  #temporary fix
        return 100, {'0':'0', '100':'100'}, [0,100], '(none)' 

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
        
    return fnmax, fnmarks, fnvalue, vfilename 

# ***************
# * run_single
# ***************
# create single prediction at first frame
@app.callback(
    [Output('model-output', 'figure')],
    [Input('button-single', 'n_clicks')],
    [State('input-dirpath', 'children'),
     State('slider-framenums','value'),
     State('slider-confidence', 'value'),
     State('cb-person','value'),
     State('cb-vehicle','value'),
     State('cb-environment','value')
     ],
)
def run_single(n_clicks, dirpath, framerange, confidence,
               cb_person, cb_vehicle, cb_environment):

    if dirpath is not None and os.path.isdir(dirpath):
        fnames = getImageFileNames(dirpath)
        imgfile = fnames[framerange[0]]
        im = Image.open(imgfile)
    else: 
        go.Figure().update_layout(title='Incorrect dirpath')
        im = Image.new('RGB',(640,480))
        fig = pil_to_fig(im, showlegend=True, title='No Image')
        return fig,

    theseObjects = [ *cb_person, *cb_vehicle, *cb_environment]
    wasConfidence = detr.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST

    if sorted(theseObjects) != sorted(detr.selObjNames) or \
       abs(confidence - wasConfidence) > 0.00001:
        detr.__init__(selectObjectNames=theseObjects, score_threshold=confidence)

    tstart = time.time()
    scores, boxes, selClasses = detect_scores_bboxes_classes(imgfile, detr)
    tend = time.time()

    fig = pil_to_fig(im, showlegend=True, title=f'DETR Predictions ({tend-tstart:.2f}s)')

    seenClassIndex = {}
    insertOrder = []
    for i,class_id in enumerate(selClasses):
        classname = CLASSES[class_id]

        if seenClassIndex.get(classname,None) is not None:
            seenClassIndex[classname] += 1
        else:
            seenClassIndex[classname] = 0

        insertOrder.append([classname + ":" + str(seenClassIndex[classname]),i])
         
    insertOrder = sorted(insertOrder, \
                         key=lambda x: (x[0].split(":")[0], int(x[0].split(":")[1])))

    for label,i in insertOrder:

        confidence = scores[i]
        x0, y0, x1, y1 = boxes[i]
        class_id = selClasses[i]

        # only display legend when it's not in the existing classes
        text = f"class={label}<br>confidence={confidence:.3f}"

        add_bbox(
            fig, x0, y0, x1, y1,
            opacity=0.7, group=label, name=label, color=COLORS[class_id], 
            showlegend=True, text=text,
        )

    return fig, 


# ***************
# run_sequence
# ***************
# Produce sequence prediction with grouping  
@app.callback(
    [Output('loading-sequence', 'children'),
     Output('signal-sequence','children')],
    [Input('button-sequence', 'n_clicks')],
    [State('input-dirpath', 'children'),
     State('slider-framenums','value'),
     State('slider-confidence', 'value'),
     State('model-output','figure'),
     State('cb-person','value'),
     State('cb-vehicle','value'),
     State('cb-environment','value'),
     State('cb-options','value'),
     State('input-dilationhwidth','value'),
     State('input-minseqlength','value')]
)
def run_sequence(n_clicks, dirpath, framerange, confidence,figure,
                 cb_person, cb_vehicle, cb_environment, 
                 cb_options, dilationhwidth, minsequencelength):

    if dirpath is not None and os.path.isdir(dirpath):
        fnames = getImageFileNames(dirpath)
    else: 
        return "", "Null:None" 
    
    selectObjects = [ *cb_person, *cb_vehicle, *cb_environment]
    useBBmasks = 'useBBmasks' in cb_options
    fillSequence = 'fillSequence' in cb_options
    
    fmin, fmax = framerange
    fnames = fnames[fmin:fmax]

    # was this a repeat?
    if len(detr.imglist) != 0:
        if fnames == detr.selectFiles:
            return "", "Null:None" 

    detr.__init__(score_threshold=confidence,selectObjectNames=selectObjects)

    vfile = compute_sequence(fnames,framerange,confidence,
                             figure, selectObjects,
                             useBBmasks, fillSequence, 
                             dilationhwidth, minsequencelength)    

    return "", f'sequencevid:{vfile}'


@cache.memoize()
def compute_sequence(fnames,framerange,confidence,
                     figure,selectObjectNames,
                     useBBmasks,fillSequence,
                     dilationhwidth, minsequencelength):
    detr.selectFiles = fnames

    staticdir = os.path.join(os.getcwd(),"static")
    detr.load_images(filelist=fnames)
    detr.predict_sequence(useBBmasks=useBBmasks,selObjectNames=selectObjectNames)
    detr.groupObjBBMaskSequence()

    obsObjectsDict = get_selected_objects(figure=figure,objGroupingDict=detr.objBBMaskSeqGrpDict)

    # filtered by object class, instance, and length 
    detr.filter_ObjBBMaskSeq(allowObjNameInstances= obsObjectsDict,
                             minCount=minsequencelength)

    # fill sequence 
    if fillSequence:
        detr.fill_ObjBBMaskSequence(specificObjectNameInstances=obsObjectsDict)

    # use dilation
    if dilationhwidth > 0:
        detr.combine_MaskSequence()
        detr.dilateErode_MaskSequence(kernelShape='el', 
                                      maskHalfWidth=dilationhwidth)

    vfile = 'sequence_' + datetime.now().strftime("%Y%m%d_%H%M%S") + ".mp4"
    if not os.environ.get("VSCODE_DEBUG"):
        detr.create_animationObject(framerange=framerange,
                                useMasks=True,
                                toHTML=False,
                                figsize=(20,15),
                                interval=30,
                                MPEGfile=os.path.join(staticdir,vfile),
                                useFFMPEGdirect=True)
    return vfile 


@app.callback(Output('sequence-output','url'),
              [Input('signal-sequence','children')],
              [State('sequence-output','url')])
def serve_sequence_video(signal,currurl):
    if currurl is None: 
        return 'static/result.mp4'
    else:
        sigtype,vfile = signal.split(":")
        if vfile is not None and \
           isinstance(vfile,str) and \
           sigtype == 'sequencevid' and \
           os.path.exists(f"./static/{vfile}"):

            # remove old file
            if os.path.exists(currurl):
                os.remove(currurl)

            # serve new file
            return f"static/{vfile}"
        else:
            return currurl 


# ***************
# run_inpaint
# ***************
# Produce inpainting results 

@app.callback(
    [Output('loading-inpaint', 'children'),
     Output('signal-inpaint','children')],
    [Input('button-inpaint', 'n_clicks')]
    )
def run_inpaint(n_clicks):
    if not n_clicks:
        return "", "Null:None"
    
    assert testContainerWrite(inpaintObj=inpaint, 
                              workDir="../data",
                              hardFail=False) , "Errors connecting with write access in containers"

    staticdir = os.path.join(os.getcwd(),"static")
    vfile = 'inpaint_' + datetime.now().strftime("%Y%m%d_%H%M%S") + ".mp4"
    performInpainting(detrObj=detr,
                      inpaintObj=inpaint,
                      workDir = "../data",
                      outputVideo=os.path.join(staticdir,vfile))

    return "", f"inpaintvid:{vfile}"


@app.callback(Output('inpaint-output','url'),
              [Input('signal-inpaint','children')],
              [State('inpaint-output','url')])
def serve_inpaint_video(signal,currurl):
    if currurl is None: 
        return 'static/result.mp4'
    else:
        sigtype,vfile = signal.split(":")
        if vfile is not None and \
           isinstance(vfile,str) and \
           sigtype == 'inpaintvid' and \
           os.path.exists(f"./static/{vfile}"):

            # remove old file
            if os.path.exists(currurl):
                os.remove(currurl)

            # serve new file
            return f"static/{vfile}"
        else:
            return currurl 


# ---------------------------------------------------------------------
if __name__ == '__main__':

    app.run_server(debug=True,host='0.0.0.0',processes=1,threaded=True)
