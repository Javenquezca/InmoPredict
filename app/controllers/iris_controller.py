import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from io import BytesIO
from PIL import Image
import base64
from fastapi import APIRouter, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pandas as pd
import os


# Crear el enrutador para las peticiones
router = APIRouter()

# Inicializar el objeto Jinja2Templates globalmente para el renderizado
templates = Jinja2Templates(directory="app/templates")

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),  '..', 'models', 'lgb_model.pkl')
model_label_C = os.path.join(os.path.dirname(os.path.abspath(__file__)),  '..', 'models', 'label_encoder_ciudad.pkl')
model_label_B = os.path.join(os.path.dirname(os.path.abspath(__file__)),  '..', 'models', 'label_encoder_barrios.pkl')
model_scaler = os.path.join(os.path.dirname(os.path.abspath(__file__)),  '..', 'models', 'scaler_model.pkl')

# Cargar el modelo desde la ruta absoluta
model = joblib.load(model_path)
label_encoder_c= joblib.load(model_label_C)
label_encoder_b= joblib.load(model_label_B)
scaler = joblib.load(model_scaler)


iris = load_iris()

def predict_iris(Ciudad: str, Barrios: str, Estrato: float, Banhos: float, Habitaciones: float, AreaConstruida: float, Piso: float, Parqueaderos: float):
    """Recibe las variables y devuelve la predicción del modelo"""

    new_data = pd.DataFrame({
    'Ciudad': [Ciudad],
    'Barrios': [Barrios],
    'Estrato': [Estrato],
    'Banhos': [Banhos],
    'Habitaciones': [Habitaciones],
    'AreaConstruida': [AreaConstruida],
    'Piso': [Piso],
    'Parqueaderos': [Parqueaderos]
})
    
    new_data['Ciudad'] = label_encoder_c.transform(new_data['Ciudad'])
    new_data['Barrios'] = label_encoder_b.transform(new_data['Barrios'])

    cols_to_normalize = ['Estrato', 'Banhos', 'Habitaciones', 'AreaConstruida', 'Piso', 'Parqueaderos']
    new_data[cols_to_normalize] = scaler.transform(new_data[cols_to_normalize])

# Make predictions
    predictions = model.predict(new_data)
    # input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    # prediction = model.predict(input_data)
    return predictions 
#     return predicted_class

# def generate_plot(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
#     """Genera una matriz de dispersión usando seaborn y marca el punto de entrada del usuario en todas las gráficas"""
#     # Convertir los datos Iris a un DataFrame para usar con seaborn
#     df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
#     df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

#     # Crear la matriz de dispersión con seaborn
#     sns.set(style="ticks")
#     g = sns.pairplot(df, hue="species", markers=["o", "s", "D"], palette="husl")

#     # Agregar el punto de entrada del usuario en todas las gráficas
#     for ax in g.axes.flatten():
#         ax.scatter(sepal_length, sepal_width, color='black', label="Entrada del Usuario", edgecolors='white', s=100, zorder=5)

#     # Leyenda
#     handles, labels = ax.get_legend_handles_labels()
#     ax.legend(handles, labels, loc='upper right')

#     # Guardar la gráfica como imagen en buffer
#     buf = BytesIO()
#     plt.savefig(buf, format="png")
#     buf.seek(0)
#     img = Image.open(buf)
#     img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    
#     # Cerrar la figura para liberar memoria
#     plt.close(g.figure)

#     return img_base64

@router.get("/", response_class=HTMLResponse)
async def form(request: Request):
    """Renderiza el formulario donde el usuario puede ingresar datos"""
    return templates.TemplateResponse("index.html", {"request": request})


@router.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, Ciudad: str = Form(...), Barrios: str = Form(...), Estrato: float = Form(...), 
                  Banhos: float = Form(...), Habitaciones: float = Form(...), AreaConstruida: float = Form(...), 
                  Piso: float = Form(...), Parqueaderos: float = Form(...)):
    try:
        # Realizar la predicción
        predicted_class = predict_iris(Ciudad, Barrios, Estrato, Banhos, Habitaciones, AreaConstruida, Piso, Parqueaderos)
        predicted_class_rounded = round(predicted_class[0])
        # Renderizar el resultado
        return templates.TemplateResponse("result.html", {"request": request, "prediction": predicted_class_rounded})
    except Exception as e:
        # Registrar el error y mostrar un mensaje claro
        print(f"Error en /predict: {e}")
        return HTMLResponse(content=f"Error interno del servidor: {e}", status_code=500)

