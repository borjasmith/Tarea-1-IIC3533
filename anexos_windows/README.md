# Anexos Windows: incisos (e) e (i)

En los Macs no se pudo ver ni limitar los threads de NumPy. En Windows sí.
Corre esto y sube los resultados. Tiempo total: 30 a 45 min de espera.

## 0. Preparar el computador

- [ ] Conecta el cargador.
- [ ] Configuración > Sistema > Energía > "Nunca" suspender.
- [ ] Cierra todos los programas.
- [ ] No uses el computador mientras corre.

## 1. Python

Abre **PowerShell** y escribe:

```powershell
python --version
```

- Si dice 3.9, 3.10, 3.11 o 3.12: sigue.
- Si no hay Python o dice 3.13: instala 3.12 desde https://www.python.org/downloads/
  y marca **"Add python.exe to PATH"** en el instalador.

## 2. Entrar a la carpeta

```powershell
cd Tarea-1-IIC3533
git checkout informe
git pull
cd anexos_windows
```

## 3. Instalar

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Si sale error de "ejecución de scripts deshabilitada":

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\venv\Scripts\Activate.ps1
```

Debe verse `(venv)` al inicio de la línea.

## 4. Verificar (1 min)

```powershell
python -c "import threadpoolctl, os, json; print('cores:', os.cpu_count()); print(json.dumps(threadpoolctl.threadpool_info(), indent=1))"
```

- Debe aparecer `"internal_api": "openblas"` y un `"num_threads"`.
- **Anota** el número de cores y el `num_threads`.
- Si sale `[]`: para y avísanos.

## 5. Experimento (e) — 5 min

```powershell
python experiment_e.py --suite --repetitions 3 --output results/computer_3/observations_e.csv
```

Mientras corre: Ctrl+Shift+Esc > Rendimiento > **captura de pantalla** del uso de CPU.

## 6. Experimento (i) — 20 a 35 min

```powershell
python experiment_i.py --repetitions 3 --output-dir results/computer_3
```

## 7. Oversubscription — 5 a 15 min

```powershell
python experiment_oversub.py --output-dir results/computer_3
```

## 8. Datos del computador

```powershell
python -c "import platform, os, numpy, sklearn, joblib, threadpoolctl, json; open('results/computer_3/metadata.txt','w').write('\n'.join(['sistema='+platform.platform(), 'procesador='+platform.processor(), 'cores_logicos='+str(os.cpu_count()), 'python='+platform.python_version(), 'numpy='+numpy.__version__, 'scikit_learn='+sklearn.__version__, 'joblib='+joblib.__version__, 'threadpoolctl='+threadpoolctl.__version__, 'blas_backend=openblas (pip)', 'threadpool_info='+json.dumps(threadpoolctl.threadpool_info()), 'N=100000', 'k=300', 'B=48'])+'\n')"
```

Anota aparte (Configuración > Sistema > Acerca de):
- Modelo de procesador
- Cores físicos
- RAM
- Versión de Windows

## 9. Subir

```powershell
cd ..
git add anexos_windows/results/computer_3
git commit -m "Resultados computador 3 (Windows)"
git push
```

Si git falla: comprime `anexos_windows\results\computer_3` y mándala por WhatsApp,
junto con la captura de pantalla y los datos del paso 8.

## Si algo falla

Manda el mensaje de error completo. Causas típicas:
- Python 3.13 (usar 3.12).
- No se ve `(venv)` al inicio de la línea (repetir paso 3).
- No estás dentro de `anexos_windows` (repetir paso 2).
