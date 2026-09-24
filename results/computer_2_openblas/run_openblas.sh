#!/bin/zsh
cd /Users/joseignaciosalas/personal/Tarea-1-IIC3533
PY=/Users/joseignaciosalas/personal/Tarea-1-IIC3533/.conda-openblas/bin/python
OUT=results/computer_2_openblas
echo "=== INICIO $(date)"
echo "=== experiment_i" && $PY /private/tmp/claude-501/-Users-joseignaciosalas-personal-Tarea-1-IIC3533/3491e44b-e900-49ec-98f6-471cf0e7babd/scratchpad/src_c2/experiment_i.py --output-dir $OUT --repetitions 5 &&
echo "=== experiment_e" && $PY /private/tmp/claude-501/-Users-joseignaciosalas-personal-Tarea-1-IIC3533/3491e44b-e900-49ec-98f6-471cf0e7babd/scratchpad/src_c2/experiment_e.py --suite --repetitions 3 --output $OUT/observations_e.csv &&
echo "=== oversub" && $PY /private/tmp/claude-501/-Users-joseignaciosalas-personal-Tarea-1-IIC3533/3491e44b-e900-49ec-98f6-471cf0e7babd/scratchpad/src_c2/experiment_oversub.py --output-dir $OUT &&
$PY -c "import numpy,platform,threadpoolctl,json;open('$OUT/metadata.txt','w').write('\n'.join([f'sistema={platform.platform()}','chip=Apple M5','cores_logicos=10','python='+platform.python_version(),'numpy='+numpy.__version__,'blas_backend=openblas 0.3.34 (conda-forge, threading_layer=openmp)','entorno=conda (miniforge), .conda-openblas','threadpool_info_default='+json.dumps(threadpoolctl.threadpool_info()),'p_max=10','repeticiones_i=5','N=100000','k=300','B=48'])+'\n')" &&
cp /private/tmp/claude-501/-Users-joseignaciosalas-personal-Tarea-1-IIC3533/3491e44b-e900-49ec-98f6-471cf0e7babd/scratchpad/src_c2/experiment_oversub.py /private/tmp/claude-501/-Users-joseignaciosalas-personal-Tarea-1-IIC3533/3491e44b-e900-49ec-98f6-471cf0e7babd/scratchpad/run_openblas.sh $OUT/ &&
echo "=== FIN $(date)" || echo "=== FALLO $(date)"
