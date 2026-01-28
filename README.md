# Implementazione di un sistema di Object Detection per video sorveglianza

Addestrare una rete (o più reti per confronto) sul dataset [Penn-Fudan Pedestrian](https://www.cis.upenn.edu/~jshi/ped_html/) (vedi esercizi lezione) per localizzare le persone nelle immagini.

La rete addestrata dovrà essere usata e applicata al dataset https://data.4tu.nl/articles/_/12714416/1 (https://wisenet.checksem.fr/#/demo) che contiene dei video di telecamere di sorveglianza di persone.

I video andranno divisi nei vari frame e andrà fatta inferenza frame-per-frame e produrre il conteggio delle persone e capire quando una zone è affollata o meno.

# Preparazione

Per lanciare il progetto è necessario creare il virtualenv e installare i pacchetti python specificati nel file `requirements.txt`

```bash
python -m venv .
source bin/activate
pip install -r requirements.txt
```

# Training

Il training avviene per mezzo dello script `train.py`. Lo script usa `argparse` per effettuare il parsing degli argomenti da riga di comando.

Gli argomenti richiesti sono:
- numero di epoche del training
- modello da addestrare (tra quelli presenti in `src/models`)
- dataset da utilizzare (`base` o `text`)

Nel file `src/utils/datasets.py` sono stati definiti due dataset (`torch.utils.data.Dataset`):
- `PennFudanDataset` ricava le bounding box a partire dalle maschere presenti nel dataset (`src/data/PennFudanPed/PedMasks`)
- `PennFudanTextDataset` ricava le bounding box mediante parsing dei file di testo presenti nel dataset (`src/data/PennFudanPed/Annotation`)

Nella directory `src/models` sono definiti i tre modelli implementati:
- ssd300_vgg16
- fasterrcnn_resnet50
- fasterrcnn_resnet50_v2

Il modello passato da riga di comando viene caricato a runtime dallo script `train.py` rendendo così più semplice l'aggiunta di nuove reti: è sufficiente infatti creare un nuovo file nella directory `models` che contenga la definizione del modello all'interno di una funzione `get_models`.

Per lanciare il training:

```bash
python src/train.py 10 fasterrcnn_resnet50 text
python src/train.py 30 ssd300_vgg16 text
```

Durante il training per mezzo del logger definito in `src/utils/cv_logger` viene loggata per ogni epoca la loss, il tempo richiesto e il numero di persone rilevate.

Al termine di ogni epoca se la loss è migliore rispetto a quella dell'epoca precedente vengono salvati i pesi del modello su file e rimosso il vecchio file dei pesi. In questo modo al termine del numero previsto di epoche di addestramento rimarrà solamente il file con i pesi migliori.

# Inferenza

L'inferenza avviene tramite lo script `src/predict.py` che si aspetta come parametri:

- il nome del modello tra quelli presenti in `src/models`
- il file contenente i pesi relativi al modello precedentemente indicato
- il path del file di input
- il path del file di output

Lo script carica a runtime la definizione del modello specificata da riga di comando dalla directory `src/models`.

L'inferenza avviene come segue:
- il video di input viene processato con `ffmpeg` (invocato tramite la libreria `plumbum`) che lo scompone in frame in formato _png_. I frame vengono salvati in una directory temporanea appositamente allocata in `/tmp` per mezzo di `tempfile.TemporaryDirectory()`
- viene fatta inferenza su ogni frame e vengono tracciate le bounding box sulle immagini che vengono salvate in un'altra sottodirectory sempre in `/tmp`
- le immagini con le bounding box vengono unite per formare il video di output sempre tramite ffmpeg

```bash
python src/predict.py fasterrcnn_resnet50 fasterrcnn_resnet50_weights.pth input.mp4 output.mp4
```

# Esecuzione su cluster

A scopo di confronto training e inferenza sono state implementate anche per essere eseguite sul cluster HPC.

A tale scopo è necessario utilizzare `apptainer` per definire delle immagini da eseguire. Le definizioni delle immagini sono contenute nei file `.def` presenti nella directory `jobs`.

Si effettua quindi il build dell'immagine apptainer:
```bash
apptainer build nomefile.sif nomefile.def
```

e infine l'accodamento dello script sullo scheduler `slurm` per mezzo del file `job` che specifica allo scheduler le risorse che si intende richiedere al cluster:

```bash
sbatch nomefile.job
```

Al termine dell'esecuzione viene rimane il file dei pesi e due altri file che contengono rispettivamente lo stdout e lo stderr dell'esecuzione.
