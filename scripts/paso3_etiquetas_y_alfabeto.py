from collections import Counter, defaultdict
from pathlib import Path
import cv2

# 1 alfabeto de 36 caracteres (0-9, a-z)
ALFABETO = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

CHAR_TO_INDEX = {char: i+1 for i, char in enumerate(ALFABETO)}
index_to_char = {i+1: char for i, char in enumerate(ALFABETO)}

EXIST = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def extraer_etiquetas_y_alfabeto(nombre_archivo: str) -> tuple[str, str, str]:
    """
    Ejemplo: '00001_AN-074_C_V.jpg'
    retorna: (prefijo = 'AN', etiqueta_cruda = '074_C_V', etiqueta_limpia = '074CV')
    """
    stem = Path(nombre_archivo).stem #quita la extension .jpg -> '00001_AN-074_C_V'
    if "_" not in stem:
        return ("Sin_guion", stem, "")

    izquierda, etiqueta_cruda = stem.split("_", 1) # '00001', 'AN-074_C_V'
    prefijo = izquierda.split("-", 1)[-1] if "-" in izquierda else izquierda # 'AN'

    #excluir caracteres difuminados y guiones bajos
    etiqueta_limpia = "".join(c for c in etiqueta_cruda.upper() if c in CHAR_TO_INDEX)
    return prefijo, etiqueta_cruda, etiqueta_limpia

def texto_a_indices (texto: str) -> list[int]:
    """
    Convierte un texto a una lista de índices según el alfabeto definido.
    """
    return [CHAR_TO_INDEX[char] for char in texto]

def indices_a_texto(indices: list[int]) -> str:
    return "".join(index_to_char[i] for i in indices if i in index_to_char)


def auditar_carpeta(carpeta: Path):
    archivos = sorted([f for f in carpeta.iterdir() if f.suffix.lower() in EXIST])
    prefijos = Counter()
    caracteres_crudos = Counter()
    proporciones_p_prefijo = defaultdict(list)
    vacias = []

    for ruta in archivos:
        prefijo, etiqueta_cruda, etiqueta_limpia = extraer_etiquetas_y_alfabeto(ruta.name)
        prefijos[prefijo] += 1
        caracteres_crudos.update(etiqueta_cruda)

        if not etiqueta_limpia:
            vacias.append(ruta.name)

        #medir prop ancho/alto de la imagen hasta 15 fotos por cada prefijo
        if len(proporciones_p_prefijo[prefijo]) < 15:
            img = cv2.imread(str(ruta))
            if img is not None:
                h, w = img.shape[:2]
                proporciones_p_prefijo[prefijo].append(w / h)

        print (f"\n== Carpeta: {carpeta} ({len(archivos)} imagenes)  ==")
        print (f"== Prefijos encontrados: {len(prefijos)}  ==")

        for pref, ratios in proporciones_p_prefijo.items():
            promedio = sum(ratios) / len(ratios)
            print(f"Prefijo: {pref} | Cantidad: {prefijos[pref]} | Promedio ancho/alto: {promedio:.2f}")

        caracteres_raros = sorted(set(caracteres_crudos.keys()) - set(ALFABETO) - {"_"})
        print(f"== Caracteres crudos encontrados: {len(caracteres_crudos)}  ==")
        print(f"== Caracteres raros encontrados: {len(caracteres_raros)}==")

        print("\nEjemplos de traducción (archivo -> cruda -> limpia -> índices CTC -> texto de vuelta):")
        for ruta in archivos[:5]:
            pref, cruda, limpia = extraer_etiquetas_y_alfabeto(ruta.name)
            indices = texto_a_indices(limpia)
            vuelta = indices_a_texto(indices)
            print(f"  {ruta.name:24s} | {cruda:9s} -> {limpia:7s} -> {str(indices):28s} -> {vuelta}")


if __name__ == "__main__":
    print(f"Total de clases CTC (C): {len(ALFABETO) + 1} (índice 0 = blanco, 1..36 = 0-9 y A-Z)")
    auditar_carpeta(Path("data/raw/imgs/train"))
    auditar_carpeta(Path("data/raw/imgs/val"))