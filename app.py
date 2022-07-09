"""
Flask-app for deploying on web-server (via heroku for example).
By calling the url where the app is deployed, the user automatically sends a request to the app,
which then returns an HTML-input-formular to specify the desired apartment-configuration.
After filling out the formular and hitting the submit-button the formular-input is sent
to the app via a POST-request. Then a buy & rent price estimation is performed,
which is based on the formular-input-data and the deposited best model
(*'model_buy.p', 'model_rent.p'*) created by *modeling.py*.
The buy- & rent-estimation-values together with associated confidence-intervalls,
which are derived from the also deposited 90%-quantile relative error of the model-test-validation,
are then returned via a new HTML-page. 

Thanks to GreekDataGuy and Ken Jee for inspiration to this module:
https://towardsdatascience.com/productionize-a-machine-learning-model-with-flask-and-heroku-8201260503d2
https://github.com/PlayingNumbers/ds_salary_proj/tree/master/FlaskAPI

@author: Michael Volk
"""

from flask import Flask, request
import pandas as pd
import numpy as np
import pickle


def load_modelConfigurations():
    """Returns loaded model configuration from file for buy and rent,
    which have been saved before by modeling.py"""
    modelConfigs = {}
    for cat in ['_buy', '_rent']:
        with open(file= 'model' + cat + '.p', mode='rb') as pickled:
            data = pickle.load(pickled)
            modelConfigs['model' + cat] = data['model']
            modelConfigs['columns_used' + cat] = data['columns_used']
            modelConfigs['test_errors' + cat] = data['test_errors']
            modelConfigs['test_errors_notAbsolute' + cat] = data['test_errors_notAbsolute']
    return modelConfigs

def load_cityCoordinates(filename = "nrwCityCoordinates.csv"):    
    """Read in created mapping-file for cityname to central-coordinates of the city.
    filename = "nrwCityCoordinates.csv" can be recreated using uncommented code
    in module 'featureEngineering.py' in section 3."""
    return pd.read_csv(filename)

def cityNamesFormular(df_city_coordinates):
    """Returns for given df_city_coordinates all
    citynames in the necessary html-format to be used in the formular-dropdown-List"""
    cities_Html = "<option>""</option>" # Empty value as standard
    for city in df_city_coordinates['City']:
        # Setting default-city
        if city == 'Aachen':
            cities_Html += "<option selected>" + city + "</option>"
        else:
            cities_Html += "<option>" + city + "</option>"
    return cities_Html

def dictToHTMLTable(dictionary, specialKeys):
    """Returns given dictionary formated as html-table with key and value as separate columns.
    For given specialKeys (=list) no corresponding values are returned"""
    dict_HTML = """<table>"""
    for key in dictionary:
        if key != "chooseLocation": #shall not be listed in html-table
            dict_HTML += """<tr><td><b>""" + key + """: </b></td>"""
            if key not in specialKeys:
                dict_HTML += """<td>""" + dictionary[key] + """</td>"""
            else:
                dict_HTML += """<td>Yes</td>"""
            dict_HTML += """</tr>"""
    dict_HTML += """</table>"""
    return dict_HTML

# FLASK-app
app = Flask(__name__)
# Route decorator of Flask which wraps below function.
@app.route('/', methods=['GET', 'POST'])
def predict():
    """
    Whenever the defined route of the Route decorator above is requested with the definied HTTP-methods ('GET' or 'Post'),
    this function gets called.'
    Returns input-html-formular or result-html-tables depending on the request-method
    ('GET' vs. 'POST') sent to the server. A 'GET' request will be performed, when
    the user sends his request direct to the server. Whereas a 'POST' request will
    be performed when the user hits the submit-button of the input-formular.
    """    
    
    # Handle the GET request (Direct request of user to server)
    if request.method == 'GET':
        
        # Load created mapping-file for cityname to central-coordinates of the city
        df_city_coordinates = load_cityCoordinates()
        
        # returns formular with the necessary input fields for the user and a submit button
        return ("""
                <HTML>
                <HEAD>
                <style>
                body {
                    margin-top: 2em;
                    margin-left: 3em;
                }
                body, table, th, td {
                    font-size: 13px;
                }
                input, select {
                    font-size: 11.5px;
                }
                h1 {font-size: 20px;}
                input.largerCheckbox {
                    width: 11.5px;
                    height: 11.5px;
                  }
                input[type=submit] {
                    width: 13em;
                    height: 4em;
                }
                </style>
                </HEAD>
                <BODY>
                
                <form id="myform" form method="POST">
                
                    <h1 style="margin-bottom: 1.25em;">Buy & Rent Price Estimator App for Apartments in North Rhine-Westphalia</h1>
                    
                    <p>Get a buy and rent price-estimation for your individual apartment-configuration in german federal state North Rhine-Westphalia.</p>
                                    
                    <p>
                    <div>Choose your input-method for location of apartment in North Rhine-Westphalia:</div>
                    <div><input type="radio" id="cn" name="chooseLocation" value="cityname" onClick="deactivateCoordinates(this.form)" checked >
                         <label for="cn"> via Cityname</label></div>
                    <div><input type="radio" id="co" name="chooseLocation" value="coordinates" onClick="deactivateCityname(this.form)">
                         <label for="co"> via Coordinates (Latitude & Longitude)</label></div>
                    </p>
                    
                    <p>
                    <table>
                      <tr>
                        <td><b>Cityname: </b></td>
                        <td><select name="Cityname">
                        """  +  cityNamesFormular(df_city_coordinates) + """
                        </select></td>
                      </tr>
                      <tr>
                        <td><b>Latitude: </b></td>
                        <td><input type="number" name="Latitude" min="50.5600" max="52.3400" step="0.0001" value="" disabled></td>
                      </tr>
                      <tr>
                        <td><b>Longitude: </b></td>
                        <td><input type="number" name="Longitude" min="6.0300" max="9.3700" step="0.0001" value="" disabled></td>
                      </tr>
                      <tr>
                        <td><b>Category: </b></td>
                        <td><select name="Category">
                            <option>Apartment</option>
                            <option>Floor-Apartment</option>
                            <option>Maisonette</option>
                            <option>Penthouse</option>
                            <option>Terrace-Apartment</option>
                            <option>Loft</option>
                        </select></td>
                      </tr>
                      <tr>
                        <td><b>Area: </b></td>
                        <td><input type="number" name="Area" min="20" max="180" step="1" value="100"></td>
                      </tr>
                      <tr>
                        <td><b>Rooms: </b></td>
                        <td><input type="number" name="Rooms" min="1" max="7" step="1" value="4"></td>
                      </tr>
                      <tr>
                        <td><b>Year: </b></td>
                        <td><input type="number" name="Year" min="1850" max="2021" step="1" value="2010"></td>
                      </tr>
                      <tr>
                        <td></td>
                      </tr>
                      <tr>
                        <td></td>
                      </tr>
                      <tr>
                        <td><b>Condition: </b></td>
                      </tr>
                      <tr>
                        <td>- First Occupancy: </td>
                        <td><input type="checkbox" class="largerCheckbox" name="First Occupancy" value="1"></td>
                      </tr>
                      <tr>
                        <td>- Upscale: </td>
                        <td><input type="checkbox" class="largerCheckbox" name="Upscale" value="1"></td>
                      </tr>
                      <tr>
                        <td>- Maintained: </td>
                        <td><input type="checkbox" class="largerCheckbox" name="Maintained" value="1" checked></td>
                      </tr>
                      <tr>
                        <td>- Renovated: </td>
                        <td><input type="checkbox" class="largerCheckbox" name="Renovated" value="1"></td>
                      </tr>
                      <tr>
                        <td>- Refurbished: </td>
                        <td><input type="checkbox" class="largerCheckbox" name="Refurbished" value="1"></td>
                      </tr>
                      <tr>
                        <td></td>
                      </tr>
                      <tr>
                        <td></td>
                      </tr>
                      <tr>
                        <td><b>Outdoor: </b></td>
                      </tr>
                      <tr>
                        <td>- Balcony: </td>
                        <td><input type="checkbox" class="largerCheckbox" name="Balcony" value="1" checked></td>
                      </tr>
                      <tr>
                        <td>- Garden: </td>
                        <td><input type="checkbox" class="largerCheckbox" name="Garden" value="1"></td>
                      </tr>
                      <tr>
                        <td>- Loggia: </td>
                        <td><input type="checkbox" class="largerCheckbox" name="Loggia" value="1"></td>
                      </tr>
                      <tr>
                        <td>- Terrace: </td>
                        <td><input type="checkbox" class="largerCheckbox" name="Terrace" value="1"></td>
                      </tr>
                    </table>
                    </p>
                    
                    <input type="submit" value="Estimate Prices!">
                    
               </form>
               
                <SCRIPT LANGUAGE="JavaScript">            
                function deactivateCoordinates(form) {
                    form.Latitude.value = ""
                    form.Longitude.value = ""
                    form.Latitude.disabled = true
                    form.Longitude.disabled = true
                    form.Cityname.disabled = false
                }
                function deactivateCityname(form) {
                    form.Cityname.value = ""
                    form.Cityname.disabled = true
                    form.Latitude.disabled = false
                    form.Longitude.disabled = false
                }
                function test1() {
                    alert("Hello")
                }
                let myform = document.getElementById('myform');
                myform.addEventListener('submit', function (evt) {
                   let error = false; 
                   if (
                           (myform.chooseLocation.value=='cityname' && myform.Cityname.value=='') || (myform.chooseLocation.value=='coordinates' && (myform.Latitude.value=='' || myform.Longitude.value=='')) || (myform.Area.value=='' || myform.Rooms.value=='' || myform.Year.value=='')
                       ) {
                       error = true
                   }
                   if (error) {
                      evt.preventDefault();
                      alert("Please fill out all necessary fields to make a Estimation!")
                   }
                });
                </SCRIPT>
               
               </BODY>
               </HTML>""")

    
    # Handle the POST request (the html-formular-input is sent via POST request after hitting submit-button by the user)
    if request.method == 'POST':
        
        # Convert the formular-data (MultiDict structure of flask), which was sent with the POST-Request, to a simple dictionary
        formular_data = request.form.to_dict()
        
        # Get 'Latitude' and 'Longitude' from formular-data depending on which input-option
        # the user has choosen (via Cityname vs. via Coordinates)
        if formular_data['chooseLocation'] == 'cityname':
            # Load created mapping-file for cityname to central-coordinates of the city
            df_city_coordinates = load_cityCoordinates()
            # Create dictionaries with each 'Latitude' and 'Longitude' and value from df_city_coordinates
            x_dict_latitude = {'Latitude': float(df_city_coordinates.loc[df_city_coordinates['City'] == formular_data['Cityname'], 'Latitude'])}
            x_dict_longitude = {'Longitude': float(df_city_coordinates.loc[df_city_coordinates['City'] == formular_data['Cityname'], 'Longitude'])}
        else:
            x_dict_latitude = {'Latitude': float(formular_data['Latitude'])}
            x_dict_longitude = {'Longitude': float(formular_data['Longitude'])}
        
        # Create dictionary which maps numerical model-features to numerical formular-data-element-names
        map_num = {'Area': 'Area',
                   'Rooms': 'Rooms',
                   'ConstructionYear': 'Year'
                  }
        # Create dictionary which maps categorical model-features to categorical formular-data-element-names
        map_cat = { 'EQ_CAT_floorApartment': ('Category', 'Floor-Apartment'),
                    'EQ_CAT_apartment': ('Category', 'Apartment'),
                    'EQ_CAT_maisonette': ('Category', 'Maisonette'),
                    'EQ_CAT_penthouse': ('Category', 'Penthouse'),
                    'EQ_CAT_terraceApartment': ('Category', 'Terrace-Apartment'),
                    'EQ_CAT_loft': ('Category', 'Loft'),
                    'EQ_CON_firstOccupancy': ('First Occupancy', '1'),
                    'EQ_CON_upscale': ('Upscale', '1'),
                    'EQ_CON_maintained': ('Maintained', '1'),
                    'EQ_CON_renovated': ('Renovated', '1'),
                    'EQ_CON_refurbished': ('Refurbished', '1'),
                    'EQ_OUT_balcony': ('Balcony', '1'),
                    'EQ_OUT_garden': ('Garden', '1'),
                    'EQ_OUT_loggia': ('Loggia', '1'),
                    'EQ_OUT_terrace': ('Terrace', '1'),
            }
        # Create dictionary with keys from map_num & map_cat and values from formular_data
        x_dict_num = {key: float(formular_data[map_num[key]]) for key in map_num}
        x_dict_cat = {key: 1 if map_cat[key] in formular_data.items() else 0 for key in map_cat}
                
        # Concatenate x_dict_num, x_dict_cat, x_dict_latitude, x_dict_longitude to x_dict
        x_dict = dict(x_dict_num)
        x_dict.update(x_dict_cat)
        x_dict.update(x_dict_latitude)
        x_dict.update(x_dict_longitude)
        
        # Load model-configuration for buy and rent dataframe
        modelConfigs = load_modelConfigurations()
        # Create new dictionary for saving results
        results = {}
        
        # Put formular data into model-configuration and make prediction for buy and rent
        for cat in ['_buy', '_rent']:
            # Create list filled with values from x_dict with order of columns_used
            results['x'+cat] = []
            for col in modelConfigs['columns_used' + cat]:
                if col in x_dict:
                    results['x'+cat].append(x_dict[col])
                else:
                    results['x'+cat].append(0)
                    print("WARNING: ", col, " not found in formular-data => value has been set to 0")
            # Convert results['x'+cat] to correct format for model-prediction
            results['x_in'+cat] = pd.DataFrame(np.array(results['x'+cat]).reshape(1,-1),
                                               columns=modelConfigs['columns_used' + cat])

            # Make prediction with model for given input-data and retransform it using np.exp()
            results['y_predicted'+cat] = np.exp(modelConfigs['model' + cat].predict(results['x_in'+cat])[0])
            
            # Calculate confidence-intervall of predicted value            
            results['y_lowerBound_5%' + cat] = (
                results['y_predicted' + cat] / np.exp(modelConfigs['test_errors_notAbsolute' + cat].quantile(q=0.95))
                )
            results['y_upperBound_5%' + cat] = (
                results['y_predicted' + cat] / np.exp(modelConfigs['test_errors_notAbsolute' + cat].quantile(q=0.05))
                )


                        
        # Return the apartment-configuration and the prediction results as 2 html-tables
        return """
                <html>
                <head>
                <style>
                body {
                    font-size: 13px;
                    margin-top: 2em;
                    margin-left: 3em;
                }
                h1 {font-size: 20px;}
                table, th, td {
                    font-size: 13px;
                    padding: 5px;
                    text-align: center;
                    border: 1px solid black;
                    border-collapse: collapse;
                }
                </style>
                </head>
                <body>
                                
                <h1>Your Configuration</h1>
                
                """ + dictToHTMLTable(formular_data, ['First Occupancy', 'Upscale', 'Maintained', 'Renovated',
                'Refurbished', 'Balcony', 'Garden', 'Loggia', 'Terrace']) + """
  
                <h1 style="margin-top: 1.5em;">Estimation Results</h1>
  
                <table>
                    <tr>
                        <th>Estimation Type</th>
                        <th>Estimated Value</th>
                        <th>Lower Bound</th>
                        <th>Upper Bound</th>
                    </tr>
                    <tr>
                        <th>Buy-price</th>
                        <td>{:,.0f} €</td>
                        <td>{:,.0f} €</td>
                        <td>{:,.0f} €</td>
                    </tr>
                    <tr>
                        <th>Buy-price per Area</th>
                        <td>{:,.0f} €/m\u00b2</td>
                        <td>{:,.0f} €/m\u00b2</td>
                        <td>{:,.0f} €/m\u00b2</td>
                    </tr>
                    <tr>
                        <th>Rent-price</th>
                        <td>{:,.0f} €</td>
                        <td>{:,.0f} €</td>
                        <td>{:,.0f} €</td>
                    </tr>
                    <tr>
                        <th>Rent-price per Area</th>
                        <td>{:,.1f} €/m\u00b2</td>
                        <td>{:,.1f} €/m\u00b2</td>
                        <td>{:,.1f} €/m\u00b2</td>
                    </tr>
                    <tr>
                        <th>Buy-to-Rent-ratio</th>
                        <td>{:,.0f}</td>
                    </tr>
                    <tr>
                        <th>Rent-to-Buy-ratio</th>
                        <td>{:,.1f}%</td>
                    </tr>
                </table>

                <h1 style="margin-top: 1.5em;">Hints</h1>
                
                <ul>
                  <li>Rent-price is shown on cold & monthly basis, while the shown Buy-to-Rent-ratio and Rent-to-Buy-ratio are based on yearly cold Rent-price</li>
                  <li>Lower and Upper Bound define a 90% Confidence Intervall for the actual price: it is expected that round about 5% of actual prices are below the Lower and round about 5% are above the Upper Bound</li>
                </ul>
                            
                <h1 style="margin-top: 1.5em;">More Information</h1>
                            
                This Buy & Rent Price Estimator App is part of my project:
                <a href="https://micvolk.github.io/Buy-and-Rent-Price-Estimator-for-Apartments">micvolk.github.io/Buy-and-Rent-Price-Estimator-for-Apartments</a> <br/>
                Visit the website and get a detailed description of the steps for building this App - starting from scraping, preparing and exploring the data from
                <a href="https://www.immowelt.de">immowelt.de</a>, evaluating different machine learning models
                and ending with transferring the best model into production by building this App.
                To see a detailed and visualised exploration of the prepared data please follow this link:
                <a href="https://micvolk.github.io/Buy-and-Rent-Price-Estimator-for-Apartments/presentation/Exploring.html">micvolk.github.io/Buy-and-Rent-Price-Estimator-for-Apartments/presentation/Exploring.html</a>
                
                <p style="margin-top: 2.5em"><em>Author: <a href="https://github.com/micvolk">Michael Volk</a></em><br>
                
                </body>
                </html>
                
                """.format(
                        results['y_predicted_buy'],
                        results['y_lowerBound_5%_buy'],
                        results['y_upperBound_5%_buy'],
                        
                        results['y_predicted_buy'] / x_dict['Area'],
                        results['y_lowerBound_5%_buy'] / x_dict['Area'],
                        results['y_upperBound_5%_buy'] / x_dict['Area'],
                        
                        results['y_predicted_rent'],
                        results['y_lowerBound_5%_rent'],
                        results['y_upperBound_5%_rent'],
                        
                        results['y_predicted_rent'] / x_dict['Area'],
                        results['y_lowerBound_5%_rent'] / x_dict['Area'],
                        results['y_upperBound_5%_rent'] / x_dict['Area'],
                        
                        results['y_predicted_buy']/(12 * results['y_predicted_rent']),
                        (12 * results['y_predicted_rent'])/results['y_predicted_buy']*100
                        )