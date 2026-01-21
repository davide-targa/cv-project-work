# Implementazione di un sistema di Object Detection per video sorveglianza

Addestrare una rete (o più reti per confronto) sul dataset [Penn-Fudan Pedestrian](https://www.cis.upenn.edu/~jshi/ped_html/) (vedi esercizi lezione) per localizzare le persone nelle immagini.

La rete addestrata dovrà essere usata e applicata al dataset https://data.4tu.nl/articles/_/12714416/1 (https://wisenet.checksem.fr/#/demo) che contiene dei video di telecamere di sorveglianza di persone.

I video andranno divisi nei vari frame e andrà fatta inferenza frame-per-frame e produrre il conteggio delle persone e capire quando una zone è affollata o meno.

# Informazioni

Per effettuare il training:

```bash
python src/train.py 10 fasterrcnn_resnet50 extended
```

Per l'inferenza:

```bash

```
