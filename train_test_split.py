import os
import random
import shutil

train_dir = "train"
validation_dir = "validation"

# Lijst van koeksoorten
koeksoorten = os.listdir(train_dir)

# Loop door de koeksoorten en voer de train/test split uit
for koeksoort in koeksoorten:
    koeksoort_dir = os.path.join(train_dir, koeksoort)
    validation_koeksoort_dir = os.path.join(validation_dir, koeksoort)
    
    # Controleer of de directory al bestaat in de validation directory
    if not os.path.exists(validation_koeksoort_dir):
        os.makedirs(validation_koeksoort_dir)
    
    # Lijst van bestanden in de koeksoort directory
    bestanden = os.listdir(koeksoort_dir)
    
    # Lijst van bestanden die al in de validation directory staan
    validation_bestanden = os.listdir(validation_koeksoort_dir)
    
    # Bereken het aantal bestanden dat verplaatst moet worden voor de test set
    n_validation = int(0.2 * len(bestanden))
    
    # Controleer of er voldoende bestanden zijn voor de test set
    if n_validation >= len(bestanden):
        raise ValueError("Niet voldoende bestanden in de directory {} voor een 80/20 split.".format(koeksoort_dir))
    
    # Bereken het aantal bestanden dat nog verplaatst kan worden
    n_available = len(bestanden) - len(validation_bestanden)
    
    # Controleer of er voldoende beschikbare bestanden zijn voor de test set
    if n_validation > n_available:
        raise ValueError("Niet voldoende beschikbare bestanden in de directory {} voor een 80/20 split.".format(koeksoort_dir))
    
    # Kies willekeurig bestanden voor de test set
    validation_bestanden += random.sample([bestand for bestand in bestanden if bestand not in validation_bestanden], n_validation)
    
    # Verplaats de gekozen bestanden naar de validation directory
    for bestand in validation_bestanden:
        bestand_src = os.path.join(koeksoort_dir, bestand)
        bestand_dst = os.path.join(validation_koeksoort_dir, bestand)
        shutil.move(bestand_src, bestand_dst)
        print("Bestand {} verplaatst naar {}".format(bestand, validation_koeksoort_dir))
