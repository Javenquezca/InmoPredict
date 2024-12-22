from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates  # Importamos RedirectResponse
from app.controllers.iris_controller import router as iris_router

# Crear la instancia de FastAPI
app = FastAPI()

# Montar cada carpeta de recursos estáticos de forma independiente
app.mount("/css", StaticFiles(directory=Path(__file__).resolve().parent / "static/css"), name="css")
app.mount("/js", StaticFiles(directory=Path(__file__).resolve().parent / "static/js"), name="js")
app.mount("/images", StaticFiles(directory=Path(__file__).resolve().parent / "static/images"), name="images")
app.mount("/fonts", StaticFiles(directory=Path(__file__).resolve().parent / "static/fonts"), name="fonts")
app.mount("/videos", StaticFiles(directory=Path(__file__).resolve().parent / "static/videos"), name="videos")


# Registrar las rutas del controlador
app.include_router(iris_router, prefix="/iris", tags=["iris"])

# Redirigir automáticamente la raíz (/) a /iris
@app.get("/")
async def root():
    # Inicializar el objeto Jinja2Templates globalmente para el renderizado
    templates = Jinja2Templates(directory="app/templates")
    return templates.TemplateResponse("home.html", {"request": {}})



#uvicorn app.main:app --reload