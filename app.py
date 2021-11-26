from flask import Flask, jsonify, request
import joblib
import sklearn


#curl -d "{\"Medidas\":[[1,2,3,4]]}" -H "Content-Type: application/json" -X POST http://127.0.0.1:5000/predecir

app= Flask(__name__)
activo= joblib.load('ml/activo-reflexivo.pkl')
globalm= joblib.load('ml/secuencial-global.pkl')
sensitivo= joblib.load('ml/sensitivo-intuitivo.pkl')
verbal= joblib.load('ml/visual-verbal.pkl')

@app.route("/")
def home():
    return "hola"
  

@app.route("/activot")
def activot():
    pregunta1=request.args.get('pregunta1')
    pregunta17=request.args.get('pregunta17')
    pregunta33=request.args.get('pregunta33')
    pregunta37=request.args.get('pregunta37')
    pregunta41=request.args.get('pregunta41')
    features=[[pregunta1,pregunta17,pregunta33,pregunta37,pregunta41]]
    
    label_index=activo.predict(features)
    return (str(round(label_index[0],2)))
@app.route("/globalt")
def globalt():
    pregunta8=request.args.get('pregunta8')
    pregunta16=request.args.get('pregunta16')
    pregunta20=request.args.get('pregunta20')
    pregunta24=request.args.get('pregunta24')
    pregunta36=request.args.get('pregunta36')
    features=[[pregunta8,pregunta16,pregunta20,pregunta24,pregunta36]]
    
    label_index=globalm.predict(features)
    return (str(round(label_index[0],2)))
@app.route("/sensitivot")
def sensitivot():
    pregunta2=request.args.get('pregunta2')
    pregunta14=request.args.get('pregunta14')
    pregunta22=request.args.get('pregunta22')
    pregunta26=request.args.get('pregunta26')
    pregunta30=request.args.get('pregunta30')
    features=[[pregunta2,pregunta14,pregunta22,pregunta26,pregunta30]]
    
    label_index=sensitivo.predict(features)
    return (str(round(label_index[0],2)))
@app.route("/verbalt")
def verbalt():
    pregunta15=request.args.get('pregunta15')
    pregunta27=request.args.get('pregunta27')
    pregunta31=request.args.get('pregunta31')
    pregunta35=request.args.get('pregunta35')
    pregunta39=request.args.get('pregunta39')
    features=[[pregunta15,pregunta27,pregunta31,pregunta35,pregunta39]]
    
    label_index=verbal.predict(features)

    return (str(round(label_index[0],2)))


if __name__ == '__main__':
    app.run(debug=True)
