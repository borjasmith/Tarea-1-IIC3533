# Anexos en Windows: incisos (e) e (i) con control real de threads

## Por qué te pedimos esto

En los dos Macs, NumPy usa la librería de matrices de Apple (Accelerate). La herramienta
`threadpoolctl`, que el enunciado pide usar en (e) e (i) para ver y limitar los threads
internos, **no reconoce Accelerate**. Resultado: en ambos Macs la lista de threads salió
vacía y el límite `t` no hizo nada.

En Windows, NumPy usa OpenBLAS, y `threadpoolctl` sí lo ve y sí lo controla. Por eso tu
computador es la única fuente que puede responder de verdad esas dos partes del enunciado.
Lo que corras aquí va al informe como un anexo dentro de (e) y de (i), titulado
"Anexo: computador 3 (Windows)".

Tiempo total estimado: **30 a 45 minutos**, casi todo de espera.

## Antes de empezar

- Conecta el cargador.
- Configuración > Sistema > Energía: pon "Nunca" en suspender la pantalla y el equipo.
- Cierra todo lo demás (navegador, Spotify, Teams). Cualquier programa abierto ensucia
  las mediciones.
- No uses el computador mientras corren los experimentos.

## Paso 1: Python

Necesitas Python 3.9, 3.10, 3.11 o 3.12 (NumPy 2.0.2 no funciona con 3.13). Para
revisar cuál tienes, abre **PowerShell** y escribe:

```powershell
python --version
```

Si no tienes Python o tienes 3.13, instala 3.12 desde https://www.python.org/downloads/
y, en el instalador, **marca la casilla "Add python.exe to PATH"**.

## Paso 2: descargar los scripts

Clona el repositorio y cámbiate a la rama `informe`:

```powershell
git clone https://github.com/borjasmith/Tarea-1-IIC3533.git
cd Tarea-1-IIC3533
git checkout informe
cd anexos_windows
```

## Paso 3: crear el ambiente e instalar las librerías

Copia y pega estas líneas en PowerShell, una por una:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Si al activar el ambiente PowerShell dice que "la ejecución de scripts está deshabilitada",
ejecuta esto una sola vez y vuelve a intentar:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Paso 4: verificar que threadpoolctl ve a OpenBLAS (1 minuto)

Este es el paso clave. Ejecuta:

```powershell
python -c "import numpy, threadpoolctl, os, json; print('cores logicos:', os.cpu_count()); print(json.dumps(threadpoolctl.threadpool_info(), indent=1))"
```

Debe aparecer una lista con al menos una entrada que diga `"internal_api": "openblas"` y
un `"num_threads"` con un número. **Anota ese número y el de cores lógicos**, van al
informe. Si la lista sale vacía `[]`, avísanos antes de seguir: algo pasó con la
instalación y no tiene sentido invertir el tiempo.

## Paso 5: correr los experimentos

Ejecuta los tres comandos en orden. Cada uno espera a que termine el anterior. Los
resultados se guardan en la carpeta `results\computer_3` dentro de `anexos_windows`.

**(e) Actividad y threads dentro de cada worker** (unos 5 minutos):

```powershell
python experiment_e.py --suite --repetitions 3 --output results/computer_3/observations_e.csv
```

Mientras corre, abre el **Administrador de tareas** (Ctrl+Shift+Esc), pestaña
Rendimiento, y toma una captura de pantalla del uso de CPU cuando estén corriendo
varios procesos. El enunciado pide observar el monitor del sistema; esa captura es la
evidencia.

**(i) Procesos y threads a la vez** (20 a 35 minutos según tu procesador):

```powershell
python experiment_i.py --repetitions 3 --output-dir results/computer_3
```

Nota: si tu procesador tiene hyperthreading, `os.cpu_count()` reporta el doble de cores
lógicos que físicos (por ejemplo 16 con 8 físicos). Está bien, el enunciado pide usar
cores lógicos. Solo anótalo.

**Demostración de oversubscription** (5 a 15 minutos). Corre combinaciones con más
threads que cores, a propósito, para mostrar el daño:

```powershell
python experiment_oversub.py --output-dir results/computer_3
```

## Paso 6: guardar los datos del computador

Ejecuta esto para que quede registrado tu hardware y las versiones:

```powershell
python -c "import platform, os, numpy, sklearn, joblib, threadpoolctl, json; open('results/computer_3/metadata.txt','w').write('\n'.join(['sistema='+platform.platform(), 'procesador='+platform.processor(), 'cores_logicos='+str(os.cpu_count()), 'python='+platform.python_version(), 'numpy='+numpy.__version__, 'scikit_learn='+sklearn.__version__, 'joblib='+joblib.__version__, 'threadpoolctl='+threadpoolctl.__version__, 'blas_backend=openblas (pip)', 'threadpool_info='+json.dumps(threadpoolctl.threadpool_info()), 'N=100000', 'k=300', 'B=48'])+'\n')"
```

Además anota a mano, en un mensaje o en el mismo archivo: **modelo de procesador, cores
físicos, memoria RAM y versión de Windows**. Lo ves en Configuración > Sistema > Acerca de.

## Paso 7: subir los resultados

```powershell
cd ..
git add anexos_windows/results/computer_3
git commit -m "Agrega resultados del computador 3 (Windows) para anexos de (e) e (i)"
git push
```

Si git te pide usuario y contraseña, usa tu cuenta de GitHub. Si algo falla, comprime la
carpeta `results\computer_3` y mándala por WhatsApp; con eso basta.

## Qué esperamos ver

- En (e): la lista de threads con `openblas` y `num_threads` igual a tus cores lógicos, en
  cada worker. Eso demuestra que cada proceso abre tantos threads como cores hay, y que
  con p procesos hay p veces más threads que cores: oversubscription.
- En (i): la fila p = 1 **ya no será plana**. Con t = 1 debe tardar claramente más que con
  t alto, porque ahora el límite sí se aplica.
- En la demostración: las combinaciones con p·t muy por encima de los cores deben tardar
  mucho más que la mejor combinación. En el Mac, 10 procesos × 10 threads tardó 147 s
  contra 2.8 s de la mejor. Esperamos algo parecido.

## Si algo falla

Copia el mensaje de error completo y mándalo. Los problemas típicos son: Python 3.13
instalado (usar 3.12), el ambiente no activado (debe verse `(venv)` al inicio de la línea
en PowerShell), o estar en la carpeta equivocada (debes estar dentro de `anexos_windows`).
