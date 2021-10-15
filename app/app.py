import io
import base64
import requests

import dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from requests.api import request

import numpy 


external_stylesheets = [
    'https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css',
    {
        'href': 'https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css',
        'rel': 'stylesheet',
        'integrity': 'sha384-BVYiiSIFeK1dGmJRAkycuHAHRg32OmUcww7on3RYdg4Va+PmSTsz/K68vbdEjh4u',
        'crossorigin': 'anonymous'
    }
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Scaleout - Image Classifier"
server = app.server

app.layout = html.Div([
    dbc.Nav(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.A('AML Classifier',
                                   href='/', className='navbar-brand')
                        ], className='navbar-header'), html.Div(
                        [
                            html.Ul(
                                [
                                    html.Li(html.A('Scaleout Systems', href='https://www.scaleoutsystems.com/')), html.Li(
                                        html.A('Github', href='https://github.com/scaleoutsystems/examples'))
                                ], className='nav navbar-nav')
                        ], className='collapse navbar-collapse')
                ], className='container')
        ], className='navbar navbar-inverse navbar-fixed-top'),
    html.Div(
        [
            html.H1('AML Classifier',
                    className='text-center'),
            html.P('Classify Images of cells',
                   className='text-center'),
            html.Hr(),
            html.Div(
                [
                    dcc.Upload(id='upload-image', children=html.Div(['Drag and Drop or ', html.A('Select File')]),
                               style={
                        'width': '98%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
                    },
                        # Allow multiple files to be uploaded
                        multiple=False
                    ),
                    html.Hr(),
                ], className='container')
        ], className='jumbotron'),
    html.H2('Classification', className='text-center'),
    dbc.Row([
        dbc.Col([
            html.Div(id='output-image-upload',
                     className="container text-center")
        ]),
        dbc.Col([
            html.Hr(),
            html.Div(id='output-image-result',
                     className="container text-center bg-info")
        ])
    ])
], className="container")


@app.callback(Output('output-image-upload', 'children'),
              Output('output-image-result', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              State('upload-image', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    print("In: update output", flush=True)

    image_html = []
    url = 'https://r5c50cce0.studio.scaleoutsystems.com/v1/models/models:predict'
    pred_res = []
    try:
        content_type, content_string = list_of_contents.split(',')

        print(content_type)
        contentb64_decode = base64.b64decode(content_string)
        file_obj = io.BytesIO(contentb64_decode)
        with open('temp.npz','wb') as fh:
            fh.write(file_obj.getbuffer())
            fh.flush()

        imgtest = numpy.load('temp.npz')
        print(numpy.shape(imgtest))
        imgtest=numpy.expand_dims(imgtest,0)
        inp = {"inputs": imgtest.tolist()}

        # If you are running locally with self signed certificate, then CHANGE the verify variable to False
        verify = True
        try:
            print("Predicting")
            res = requests.post(url, json=inp)
        except Exception as e:
            print(e)
        res_dict = res.json()['outputs']

        #image_html = html.Img(
        #    src='data:image/png;base64,{}'.format(content_string), width="50%")
        class_names = ['Basophil',
                       'Erythroblast',
                       'Eosinophil',
                       'Smudge cell',
                       'Lymphocyte (atypical)',
                       'Lymphocyte (typical)',
                       'Metamyelocyte',
                       'Monoblast',
                       'Monocyte',
                       'Myelocyte',
                       'Myeloblast',
                       'Neutrophil (band)',
                       'Neutrophil (segmented)',
                       'Promyelocyte (bilobled)',
                       'Promyelocyte',
                       'Total']
        classification = numpy.argmax(res.json()['outputs'])

        pred_res.append(html.Div(
            [
                #html.B(key, style={'font-weight': 'bold'}),
                html.Span(["Prediction: {} ({})".format(class_names[classification],res_dict[0][classification])])
            ], style={'font-size': '20px'}))

    except Exception as err:
        print("No image.")
        print(err)

    return image_html, pred_res


if __name__ == '__main__':
    app.run_server(debug=True)