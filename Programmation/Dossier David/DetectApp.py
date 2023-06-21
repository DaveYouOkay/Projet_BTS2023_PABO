#***********************************************************
# Projet : Projet - Prévention Alerte Bébé Oublié
# Auteur : Bezin David
# Nom du Fichier : DetectApp.py
# Date de Création : 06/06/2023
# Date de Modification : 13/06/2023
#***********************************************************
# Description : Le programme utilise la camera raspberry pi
# ainsi qu'un algorithme de reconnaissance faciale et d'objet pour determiner
# la presence d'un animal et/ou l'age d'un individu devant la camera
#
# Le programme capture les conditions environnementales pour les afficher
# Toutes les données sont stocker dans une base de données SQLite
#***********************************************************

# --- Import ---
import tkinter as tk
from tkinter import *
import sys
import cv2
from PIL import Image, ImageTk
from picamera import PiCamera
from picamera.array import PiRGBArray
import numpy as np
import bme280
import sqlite3
import time
from datetime import datetime

# -------------- WEIGHTS --------------

# ------- AGE -------
# The model architecture for age estimation
AGE_MODEL = 'face_age_weights/deploy_age.prototxt' # download from: https://drive.google.com/open?id=1kiusFljZc9QfcIYdU2s7xrtWHTraHwmW
# The model pre-trained weights for age estimation
AGE_PROTO = 'face_age_weights/age_net.caffemodel' # download from: https://drive.google.com/open?id=1kWv0AjxGSN0g31OeJa02eBGM0R_jcjIl
# Chaque Caffe Model impose la forme de l'image d'entrée et un prétraitement de l'image est nécessaire,
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# Represente les 8 classes d'age de cette couche de probabilité CNN (réseau de neurones convolutifs)
AGE_INTERVALS = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)','(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']

# ------- FACE -------
# The model architecture for face detection
FACE_MODEL = "face_age_weights/res10_300x300_ssd_iter_140000_fp16.caffemodel" # download from: https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
# The model pre-trained weights for face detection
FACE_PROTO = "face_age_weights/deploy.prototxt" # download from: https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt

# ------- ANIMAL -------
# The model architecture for animal detection
ANIMAL_MODEL = "cat-dog-weights/MobileNetSSD_deploy.caffemodel"
# The model pre-trained weights for animal detection
ANIMAL_PROTO = "cat-dog-weights/MobileNetSSD_deploy.prototxt"

TYPE_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                "sofa", "train", "tvmonitor"] #cat , dog, person

# ------- CHARGE NET -------
# Charge face Caffe model
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
# Charge age prediction model
age_net = cv2.dnn.readNetFromCaffe(AGE_MODEL, AGE_PROTO)
# Charge animal prediction model
animal_net = cv2.dnn.readNetFromCaffe(prototxt=ANIMAL_PROTO, caffeModel=ANIMAL_MODEL)
        
frame_width = 640
frame_height = 480



class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Application PABO - Détection environnementale et vidéo en direct")
        self.minsize(1080, 720)

        self.grid_columnconfigure(0, weight=1)  # Colonne 0 s'agrandit
        self.grid_columnconfigure(1, weight=0)  # Colonne 1 conserve sa taille

        self.cadre_gauche = Frame(self, width=400, height=625, bg='grey')
        self.cadre_gauche.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.cadre_gauche.grid_propagate(False)

        self.cadre_droit = Frame(self, width=640, height=480, bg='grey')
        self.cadre_droit.grid(row=0, column=1, padx=10, pady=5, sticky="e")

        self.date_label = tk.Label(self.cadre_gauche, text="Date :", font=("Helvetica", 20))
        self.date_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="w")
        
        self.heure_label = tk.Label(self.cadre_gauche, text="Heure :", font=("Helvetica", 20))
        self.heure_label.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="w")

        self.temperature_label = tk.Label(self.cadre_gauche, text="Température : ", font=("Helvetica", 16))
        self.temperature_label.grid(row=2, column=0, padx=10, pady=10, sticky="w")

        self.pression_label = tk.Label(self.cadre_gauche, text="Pression : ", font=("Helvetica", 16))
        self.pression_label.grid(row=3, column=0, padx=10, pady=10, sticky="w")

        self.humidite_label = tk.Label(self.cadre_gauche, text="Humidité : ", font=("Helvetica", 16))
        self.humidite_label.grid(row=4, column=0, padx=10, pady=10,sticky="w")
        
        self.alerte_environnement_label = tk.Label(self.cadre_gauche, text="Alerte Environnementale: ", font=("Helvetica", 16))
        self.alerte_environnement_label.grid(row=5, column=0, padx=10, pady=10,sticky="w")

        self.alerte_detection_label = tk.Label(self.cadre_gauche, text="Alerte Detection: ", font=("Helvetica", 16))
        self.alerte_detection_label.grid(row=6, column=0, padx=10, pady=10, sticky="w")

        self.extra_label = []
        
        self.extra_label.append(tk.Label(self.cadre_droit, text="-- Label supplémentaire --"))
        self.extra_label[0].grid(row=1, column=0, padx=10, pady=10)
        self.extra_label.append(tk.Label(self.cadre_droit, text="-- Label supplémentaire --"))
        self.extra_label[1].grid(row=2, column=0, padx=10, pady=10)
        self.extra_label.append(tk.Label(self.cadre_droit, text="-- Label supplémentaire --"))
        self.extra_label[2].grid(row=3, column=0, padx=10, pady=10)

        self.extra_label.append(tk.Label(self.cadre_droit, text="-- Label supplémentaire --"))
        self.extra_label[3].grid(row=1, column=1, padx=10, pady=10)
        self.extra_label.append(tk.Label(self.cadre_droit, text="-- Label supplémentaire --"))
        self.extra_label[4].grid(row=2, column=1, padx=10, pady=10)
        self.extra_label.append(tk.Label(self.cadre_droit, text="-- Label supplémentaire --"))
        self.extra_label[5].grid(row=3, column=1, padx=10, pady=10)

        self.quit_button = tk.Button(self, text="Quitter", command=self.quit_application)
        self.quit_button.grid(row=1, column=1, columnspan=2, pady=10)

        self.new_mesure_button = tk.Button(self, text="Nouvelle mesure", command=self.nouvelle_mesure)
        self.new_mesure_button.grid(row=1, column=0, pady=10)
        
        self.video_label = tk.Label(self.cadre_droit)
        self.video_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10)
        
        
    # Execution des programme nécessaire pour la 1ere fois
    def start(self):
        self.start_video_stream()
        time.sleep(1)
        self.update_environnemental_data()


    # Démarre la capture video
    def start_video_stream(self):
        self.camera = PiCamera()
        self.camera.resolution = (frame_width, frame_height)
        self.camera.framerate = 10
        self.rawCapture = PiRGBArray(self.camera, size=(640, 480))
        time.sleep(0.2)
        # Démarrer la capture de la vidéo en continu
        self.video_streaming = True
        self.capture_video_frame()


    # Traite la nouvelle frame capturé
    def capture_video_frame(self):
        if self.video_streaming:
            # Capturer une frame depuis la caméra
            self.camera.capture(self.rawCapture, format="bgr", use_video_port=True)
            frame = self.rawCapture.array

            # Traiter et afficher la frame
            frame = self.detection_all(frame)
            image = Image.fromarray(frame)
            image = ImageTk.PhotoImage(image)
            self.video_label.configure(image=image)
            self.video_label.image = image

            # Réinitialiser la frame
            self.rawCapture.truncate(0)
            # Planifier la prochaine capture de frame
            self.after(1000, self.capture_video_frame)


    # Fonction Appelant les différente détection (visage, age, forme) et la fonction de sauvegarde
    def detection_all(self,frame):
        time.sleep(0.2)
        print("== Detection de visage ==")
        faces = self.get_faces(frame)
        time.sleep(0.2)
        age_declare, frame = self.estimate_age(frame, faces)

        time.sleep(0.2)
        print("== Detection de forme ==")
        object_detected, frame = self.animal_detection(frame)
        print(object_detected)

        print("== Resume detection ==")
        self.detection_data_db(age_declare, object_detected)
        
        return frame


    # Enregistre les données de detection dans une base de données SQLite
    def detection_data_db(self, age_detected, object_detected):
        # Création d'un tableau pour y stocker les "choses" detectées : personne, chien, chat, enfant
        something_detected = []
        
        for i in range(len(age_detected)//2):
            index = i * 2
            if index + 1 < len(age_detected):
                age_label = "Visage entre " + str(age_detected[index]) + " | confiant à : " + str(round(age_detected[index + 1] * 100, 4)) + "%"
                self.extra_label[i].config(text=age_label)
                
                something_detected.append(age_detected[index])

        for i in range(len(object_detected)//2):
            index = i * 2
            if index + 1 < len(object_detected):
                object_label = "Type : " + str(object_detected[index]) + " | confiant à : " + str(round(object_detected[index + 1] * 100, 4)) + "%"
                self.extra_label[3+i].config(text=object_label)

                something_detected.append(object_detected[index])

        # Tableau Résumé de la détection
        resume_detection = self.security_data_detection(something_detected)
        print(resume_detection)
        
        # Enregistrer l'heure de détection et les données dans la base de données
        date_detection = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        number_things_detected = resume_detection[0] + resume_detection[1]
        personne_detected = resume_detection[0]
        baby_animal_detected = resume_detection[1]
        
        self.alerte_detection_label.config(text="Alerte Detection : {}".format(resume_detection[2]))

        connexion_db = sqlite3.connect("detection_data.sqlite")
        detection_db = connexion_db.cursor()
        # si n'existe pas créer une table pour stocker les informations
        detection_db.execute("CREATE TABLE IF NOT EXISTS detection(id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, number_things_detected INTEGER, personne_detected INTEGER, baby_animal_detected INTEGER)")
        detection_db.execute("INSERT INTO detection (date, number_things_detected, personne_detected, baby_animal_detected) VALUES (?, ?, ?, ?)", (date_detection, number_things_detected, personne_detected, baby_animal_detected))
        connexion_db.commit()

        if resume_detection[2] > 0:
            print("*== == == Envoie Alerte == == ==*")
            detection_db.execute("CREATE TABLE IF NOT EXISTS alerte (id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, alerte INTEGER)")
            connexion_db.commit()
            detection_db.execute("INSERT INTO alerte (date, alerte) VALUES (?, ?)", (date_detection, resume_detection[2]))
            connexion_db.commit()


    # Verifie le resultat de sécurité de la détection et renvoie un tableau résumé
    def security_data_detection(self, something_detected):
        person = 0
        vulnerable = 0
        alerte = 0
        resume_detection = []
        for i in range(len(something_detected)):
            if something_detected[i] == "cat" or something_detected[i] == "dog":
                vulnerable = vulnerable + 1
                
            for age_enfant in range(len(AGE_INTERVALS) - 5):
                if something_detected[i] == AGE_INTERVALS[age_enfant]:
                    vulnerable = vulnerable + 1
                
            for age_adult in range(len(AGE_INTERVALS) - 3):
                if something_detected[i] == AGE_INTERVALS[age_adult+3]:
                    person = person + 1
                                
        if vulnerable > 0 and person == 0:
            alerte = 1
        
        resume_detection.extend([person, vulnerable, alerte])
        
        return resume_detection


    # Verifie le resultat de sécurité de l'environnement et renvoie le nombre d'alerte detecté
    def security_data_environnement(self, temperature, pression, humidite):
    #https://mobile.interieur.gouv.fr/Archives/Archives-de-la-rubrique-Ma-securite/Avec-votre-vehicule/Chaleur-quels-reflexes-adopter-en-voiture-avec-des-enfants
    # >1500m altitude enfant = début diffculté respiratoire enfant ; 1500m = 850hPa 
        nombreAlerte = 0
        if(temperature > 30.0 or temperature < 18.0):
            print("Alerte temperature ")
            nombreAlerte = nombreAlerte + 1 
        if(pression < 850.0):
            print("Alerte pression")
            nombreAlerte = nombreAlerte + 1
        if(humidite > 80.0 or humidite < 35.0):
            print("Alerte humidite")
            nombreAlerte = nombreAlerte + 1

        return nombreAlerte


    # Enregistre les données environnementale dans une base de données SQLite
    def data_environnement_db(self, temperature, pression, humidite):
        connexion_db = sqlite3.connect("detection_data.sqlite")
        bme_db = connexion_db.cursor()
        # si n'existe pas créer une table pour stocker les informations capté par le BME 
        bme_db.execute("CREATE TABLE IF NOT EXISTS environnement(id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, temperature REAL, pression REAL, humidite REAL)")

        # Enregistrer l'heure de détection et les données dans la base de données
        date_detection = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        bme_db.execute("INSERT INTO environnement (date, temperature, pression, humidite) VALUES (?, ?, ?, ?)", (date_detection, temperature, pression, humidite))
        connexion_db.commit()
        print("Donnees environnement enregistre dans la base de donnees")

        # Affiche les données sur l'IHM
        dateNow = datetime.now()
        mois_dic = ["janvier", "fevrier", "mars", "avril","mai","juin","juillet","aout","septembre","octobre", "novembre", "decembre"]
        date_text = str(dateNow.day) +" "+ mois_dic[dateNow.month - 1] + " "+ str(dateNow.year)
        heure_text = str(dateNow.hour) +" h: "+ str(dateNow.minute) +" m: "+ str(dateNow.second) +" s"     
        
        self.date_label.config(text="Date : {}".format(date_text))
        self.heure_label.config(text="Heure : {}".format(heure_text))
        self.temperature_label.config(text="Température : {}°C".format(temperature))
        self.pression_label.config(text="Pression : {}hPa".format(pression))
        self.humidite_label.config(text="Humidité : {}%".format(humidite))
        
        alerte_environnement = self.security_data_environnement(temperature, pression, humidite)
        self.alerte_environnement_label.config(text="Alerte Environnementale : {}".format(alerte_environnement))

        if alerte_environnement > 0:
            print("*== == == Envoie Alerte == == ==*")
            bme_db.execute("CREATE TABLE IF NOT EXISTS alerte (id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, alerte INTEGER)")
            connexion_db.commit()
            bme_db.execute("INSERT INTO alerte (date, alerte) VALUES (?, ?)", (date_detection, alerte_environnement))
            connexion_db.commit()


    # Récupère les données environnementale et s'execute toute les 10sec
    def update_environnemental_data(self):
        print("== Reception données environnementales ==")
        #creer des variables et y insere les donnees du bme280
        bmeTemperature, bmePression, bmeHumidite = bme280.readBME280All()
    
        #arrondie des donnees
        temperature = round(bmeTemperature, 1)
        pression = round(bmePression,1)
        humidite = round(bmeHumidite,1)
        #
        self.data_environnement_db(temperature, pression, humidite)

        self.after(10000, self.update_environnemental_data)


    # Détecte la présence de chien et chat
    def animal_detection(self,frame):
        (H, W) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        animal_net.setInput(blob)
        animal_detections = animal_net.forward()

        detection_valid = 0
        important_values = []
        for i in np.arange(0, animal_detections.shape[2]):
            confidence = animal_detections[0, 0, i, 2]
            if confidence > 0.63: # n'affiche un type que si confiance supérieur
                idx = int(animal_detections[0, 0, i, 1])

                if TYPE_CLASSES[idx] != "dog" and TYPE_CLASSES[idx] !="cat": #and TYPE_CLASSES[idx] !="person":
                    continue

                animal_box = animal_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (start_x, start_y, end_x, end_y) = animal_box.astype("int")
                #=========
                detection_valid = detection_valid + 1
                print("="*5, f"Detection {detection_valid} Prediction Probabilities", "="*5)
                i = animal_detections[0].argmax()
                type_race = TYPE_CLASSES[idx]
                type_confidence_score = confidence
                # Dessine la boite
                label = f"Type:{type_race} - {type_confidence_score*100:.2f}%"
                print(label)
                # Obtient la position où mettre le texte
                yPos = start_y - 15
                while yPos < 15:
                    yPos += 15
                # écrit le text dans la frame
                frame = cv2.putText(frame, label, (start_x, yPos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness=2)
                frame = cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
            
                important_values.append(type_race)
                important_values.append(round(type_confidence_score,4))

        return important_values, frame


    # Identifie des visage et retourne leur position
    def get_faces(self,frame):
        # Un blob est essentiellement un tenseur multidimensionnel (tableau de valeurs) qui représente l'image
        # convertir la frame en un blob prêt pour l'entrée dans le Reseau Neuronal
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177.0, 123.0))
    
        # définir l'image comme entrée du RN
        face_net.setInput(blob)
        # effectuer une inférence et obtenir des prédictions
        output = np.squeeze(face_net.forward())
        # initialise la liste de resultat
        faces = []
        # boucle sur les visages détectés
        for i in range(output.shape[0]):
            confidence = output[i, 2]
            if confidence > 0.5:
                box = output[i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                # converti en entier
                start_x, start_y, end_x, end_y = box.astype(np.int)
                # élargi un peu la boîte
                start_x, start_y, end_x, end_y = start_x - \
                    10, start_y - 10, end_x + 10, end_y + 10
                start_x = 0 if start_x < 0 else start_x
                start_y = 0 if start_y < 0 else start_y
                end_x = 0 if end_x < 0 else end_x
                end_y = 0 if end_y < 0 else end_y
                # ajouter à la liste
                faces.append((start_x, start_y, end_x, end_y))
        return faces


    # Estime l'age des visage detecté par get face
    def estimate_age(self, frame, faces):
        important_values = []
        for i, (start_x, start_y, end_x, end_y) in enumerate(faces):
                face_img = frame[start_y: end_y, start_x: end_x]
                # image --> Input image pour preprocess avant de la passer dans le dnn pour la classification.
                blob = cv2.dnn.blobFromImage(
                    image=face_img, scalefactor=1.0, size=(227, 227), 
                    mean=MODEL_MEAN_VALUES, swapRB=False
                )
                # Estime Age
                age_net.setInput(blob)
                age_prediction = age_net.forward()
                print("="*5, f"Face {i+1} Prediction Probabilities", "="*5)
                #for i in range(age_prediction[0].shape[0]):
                #    print(f"{AGE_INTERVALS[i]}: {age_prediction[0, i]*100:.2f}%")
                age_in_tab = age_prediction[0].argmax()

                age = AGE_INTERVALS[age_in_tab]
                age_confidence_score = age_prediction[0][age_in_tab]
            
                # Dessine la boite
                label = f"Age:{age} - {age_confidence_score*100:.2f}%"
                print(label)
            
                # Obtient la position où mettre le texte
                yPos = start_y - 15
                while yPos < 15:
                    yPos += 15
                # écrit le text dans la frame
                frame = cv2.putText(frame, label, (start_x, yPos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)
                # Dessine le rectangle autour de la tête
                frame = cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color=(255, 0, 0), thickness=2)
            
                important_values.append(age)
                important_values.append(round(age_confidence_score,4))
        
        return important_values, frame


        # Actualise les mesures environnementales au clique d'un bouton
    def nouvelle_mesure(self):
        self.update_environnemental_data()


    # Bouton pour fermer le programme
    def quit_application(self):
        self.camera.close()
        self.destroy()
        cv2.destroyAllWindows()
        raise SystemExit


# créer un objet de la classe Application et lance la fonction start() de l'objet app
if __name__ == "__main__":
    app = Application()
    app.start()