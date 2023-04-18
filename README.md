## **Plant Leaf Disease Detection using Machine Learning**

A system which allows the user to upload image of a plant leaf and predict if it is disease affected;
- Demo Video - Web Application:  https://drive.google.com/file/d/1X38fXQnTdRsdOka1jJW9CoURDNNgc_V5/view?usp=sharing
- Demo Video - Android Application: https://drive.google.com/file/d/1B_JmcuyiOlUoGolBFFUnIN9XWqElXcH7/view?usp=sharing
- Project Report: https://drive.google.com/file/d/134S-2Es4jajlHdyalZgmlbw77HOLt8VC/view?usp=sharing
- Project Paper: https://drive.google.com/file/d/1P3HJ5hr53-1KexXE0dG7CdUjzEPb7z5g/view?usp=sharing

#### Plants currently identified:
- Maize 
- Corn

### Machine Learning Model
 -  Used Convolution Neural Networks
 -  Used Tensorflow for builiding the CNN
 
### Rest API's Created
#### Used pickle to save the ML model and Flask to provide the Frontend as well the API's.

API Route : https://plantdiseaseash.herokuapp.com/train or https://plantdiseaseash.herokuapp.com/test
Method: GET
Action: Trains the ML model and return the training metrics which can be further used for visualisation.


| Method | API ROUTE | Actions |
|--|--| --|
| GET |https://plantdiseaseash.herokuapp.com/test or https://plantdiseaseash.herokuapp.com/test  | Trains the ML model and return the training metrics which can be further used for visualisation |
| POST |https://plantdiseaseash.herokuapp.com/predict | Image to be tested is uploaded via a POST request and the predictions are returned. |
| GET |https://plantdiseaseash.herokuapp.com/getAllDiseases | Returns a JSON object containing list of disease classes. |





### Hosted on Heroku.com
- Main Prediction Page: [https://plantdiseaseash.herokuapp.com/home](https://plantdiseaseash.herokuapp.com/home)
- Model Training Visualization : [https://plantdiseasetrain.herokuapp.com/](https://plantdiseasetrain.herokuapp.com/)

