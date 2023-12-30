import pandas as pd
from sklearn import preprocessing
import unidecode
from collections import defaultdict
import math
import random
import numpy as np
from tqdm import tqdm
import os
import datetime
import polars as pl 
from polars import col, lit, when 


import warnings

warnings.filterwarnings('ignore')

DATASET_GESTIONALE = False

comorbidita_dict = {
    "OBIESITA": "OBESITA",
    "OBIESITÀ": "OBESITA",
    "OBIESITA'": "OBESITA",
    "OBESITA": "OBESITA",
    "OBESITÀ'": "OBESITA",
    "IPERCOLESTREOLEMIA": "IPERCOLESTEROLEMIA",
    "OBESITA'": "OBESITÀ",
    "CARDIOPATRIA ISCHIMICA CRONICA": "CARDIOPATIA ISCHEMICA CRONICA",
    "IPERTENSIONE ARTERIOSO": "IPERTENSIONE ARTERIOSA",
    "IPOTIERODISMO": "IPOTIROIDISMO",
    "IPERPLASSIA SURENALICA": "IPERPLASIA SURRENALICA",
    "ADENOMA PROSTATICA": "ADENOMA PROSTATICO",
    "S. DEPRESSIVA": "SINDROME DEPRESSIVA",
    "ENDOMETROSI": "ENDOMETRIOSI",
    "SINDROME ANSIOSO-DEPRESSIVA": "SINDROME DEPRESSIVA",
    "PROGRESO TVP": "PREGRESSO TVP",
    "OVAIA POLICISTICA": "OVAIO POLICISTICO",
    "ADENOMA MAMMARIA": "ADENOMA MAMMARIO",
    "LINFOMA GASTRICA": "LINFOMA GASTRICO",
    "STENT ANEURISMO": "STENT ANEURISMA",
    "DISPLEDIMIA": "DISLIPIDEMIA",
    "PROGRESSO INFARTO MIOCARDICO": "PREGRESSO INFARTO MIOCARDIO",
    "BPCO ASMATIFORMR": "BPCO ASMATIFORME",
    "PRORESSO TIA": "PREGRESSO TIA",
    "PROGRESSO TIA": "PREGRESSO TIA",
    "CARDIOPATIA ISCHIMICA": "CARDIOPATIA ISCHEMICA",
    "DISLIPEDEMIA": "DISLIPIDEMIA",
    "DISPLIDEMIA": "DISLIPIDEMIA",
    "DISLIPOÌIDEMIA": "DISLIPIDEMIA",
    "PROGRESSO INFARTO MIOCARDIO": "PREGRESSO INFARTO MIOCARDIO",
    "ANEMIA MEDITERRAN": "ANEMIA MEDITERRANEA",
    "SECONDO": "II",
    "PREGRESSO INFARTO DEL MIOCARDIO": "PREGRESSO INFARTO MIOCARDIO",
    "GLAUCOM": "GLAUCOMA",
    "DEPRESSIONE": "SINDROME DEPRESSIVA",
    "IPERTENSIONE ARTERIOAS": "IPERTENSIONE ARTERIOSA",
    "NOSULO TIROIDEO": "NODULO TIROIDEO",
    "S.DEPRESSIVA": "SINDROME DEPRESSIVA",
    "S. ANSIOSO-DEPRESSIVA": "SINDROME DEPRESSIVA",
    "IPERTENSIONE AARTERIOSA": "IPERTENSIONE ARTERIOSA",
    "DETERIORMNTO COGNITIVO": "DETERIORAMENTO COGNITIVO",
    "IPERTENSIONE RTERIOSA": "IPERTENSIONE ARTERIOSA",
    "GLAUCOMAA": "GLAUCOMA",
    "GLAUCOME": "GLAUCOMA",
    "IPOTIROISIMO": "IPOTIROIDISMO",
    "IPOTIROIDIOSMO": "IPOTIROIDISMO",
    "DISLPIDEMIA": "DISLIPIDEMIA",
    "VARICI ART .INF": "VARICI ARTI INF",
    "TRAID TALASIMICP": "TRAIT TALASSEMICO",
    "DISLIDEMIA": "DISLIPIDEMIA",
    "DISLIPEDIMIA": "DISLIPIDEMIA",
    "ARITMIE": "ARITMIA",
    "DISLIPIEDEMIA": "DISLIPIDEMIA",
    "DISLIPIDEMIA ": "DISLIPIDEMIA",
    "DISLIPIDIMIA": "DISLIPIDEMIA",
    "DISTURDO SCHIZOTIPICO": "DISTURBO SCHIZOTIPICO",
    "FUMO ": "FUMO",
    "INTOLLERANZA GLICIDICA": "INTOLLERANZA GLUCIDICA",
    "IPERTENSIONE ARTERIOSA ": "IPERTENSIONE ARTERIOSA",
    "IPOTIROIDISMO ": "IPOTIROIDISMO",
    "IPOTIRPIDISMO": "IPOTIROIDISMO",
    "S.DEPRISSIVA": "SINDROME DEPRESSIVA",
    "SD ANSIOSA": "SINDROME ANSIOSA",
    "FIBROMIALGIA ": "FIBROMIALGIA",
    "K.POLMONARE": "K POLMONARE",
    "S. ANSIOSA": "SINDROME ANSIOSA",
    "ANSIA": "SINDROME ANSIOSA",
    "CARDIOPATIA DILATATIVA": "CARDIOMIOPATIA DILATATIVA",
    "EPATOPATIA HCV RELATA": "EPATOPATIA HCV CORRELATA",
    "IOTIROIDISMO": "IPOTIROIDISMO",
    "IPEURICEMIA": "IPERURICEMIA",
    "IPOTIROIDSIMO": "IPOTIROIDISMO",
    "OIPOTIROIDISMO": "IPOTIROIDISMO",
    "POLMONITE DA COVID": "POLMONITE COVID",
    "PREGRESSO INFARTO MIOCARDICO": "PREGRESSO INFARTO MIOCARDIO",
    "TRAPIANTO DI RENE": "TRAPIANTO RENALE"
}

terapia_pre_dict = {
    "AMILODIPINA": "AMLODIPINA",
    "ALLUPRONOLOLO": "ALLOPURINOLO",
    "BISOPROLO": "BISOPROLOLO",
    "DELTACORTONE": "DELTACORTENE",
    "DILATERND": "DILATREND",
    "DILATRENT": "DILATREND",
    "EUTIROPX": "EUTIROX",
    "INDURAL": "INDERAL",
    "LASIXX": "LASIX",
    "MICONFENOLATTO": "MICONFENOLATO",
    "MIRTAAPINA": "MIRTAZAPINA",
    "NATCAL": "NATECAL",
    "NEBIVILOLO": "NEBIVOLOLO",
    "OMONIC": "OMNIC",
    "OMREPAZOLO": "OMEPRAZOLO",
    "PANTAPRAZOLO": "PANTOPRAZOLO",
    "QUETAPINA": "QUETIAPINA",
    "SERATOLINA": "SERTRALINA",
    "SSALBUTAMOLO": "SALBUTAMOLO",
    "STATINA": "STATINE",
    "TARDIFER": "TARDYFER",
    "TMSULOSINA": "TAMSULOSINA",
    "URORETIC": "UROREC",
    "VELPTTORO": "VELPHORO",
    "NO": "NO_TERAPIA"
}

lateralit_pre_dict = {
    "0.0": "LATERALITA_NESSUNA",
    "1.0": "LATERALITA_DESTRA",
    "2.0": "LATERALITA_SINISTRA",
    "3.0": "LATERALITA_BILATERALE",

}

mallampati_pre_dict = {
    "1": "MALLAMPATI_1",
    "2": "MALLAMPATI_2",
    "3": "MALLAMPATI_3",
    "4": "MALLAMPATI_4",
    "5": "MALLAMPATI_5",
}

asa_pre_dict = {
    "1": "ASA_1",
    "2": "ASA_2",
    "3": "ASA_3",
    "4": "ASA_4",
    "5": "ASA_5",
}

catetere_vescicale_pre_dict = {
    "NO": "CATETERE_VESCIALE_NO",
    "SI": "CATETERE_VESCIALE_SI",
    "no": "CATETERE_VESCIALE_NO",
}

cvc_pre_dict = {
    "NO": "CVC_NO",
    "SI": "CVC_SI",
    "no": "CVC_NO"
}

reintervento_pre_dict = {
    "NO": "REINTERVENTO_NO",
    "SI": "REINTERVENTO_SI",
    "no": "REINTERVENTO_NO",
}

accesso_chirurgico_predict = {
    "1": "OPEN",
    "2": "LAPAROSCOPIA",
    "3": "ENDOSCOPIA",
    "4": "TORACOSCOPIA",
    "5": "ROBOTICA",
    "6": "PERCUTANEO"
}

diabete_mellito_predict = {
    "0.0": "NO_DIABETE_MELLITO",
    "1.0": "DIABETE_MELLITO_1",
    "2.0": "DIABETE_MELLITO_2",
    "3.0": "DIABETE_MELLITO_3",
    "4.0": "DIABETE_MELLITO_4",
    "5.0": "DIABETE_MELLITO_5",
    "6.0": "DIABETE_MELLITO_6"
}

diagnosis = {
    (1, 139): "Malattie infettive e parassitarie",
    (140, 239): "Tumori",
    (240, 279): "Malattie delle ghiandole endocrine, della nutrizione e del metabolismo, e disturbi immunitari",
    (280, 289): "Malattie del sangue e organi emopoietici",
    (290, 319): "Disturbi Mentali",
    (320, 389): "Malattie del sistema nervoso e degli organi di senso",
    (390, 459): "Malattie del sistema circolatorio",
    (460, 519): "Malattie dell’apparato respiratorio",
    (520, 579): "Malattie dell’apparato digerente",
    (580, 629): "Malattie dell’apparato genitourinario",
    (630, 677): "Complicazioni della gravidanza, del parto e del puerperio",
    (680, 709): "Malattie della pelle e del tessuto sottocutaneo",
    (710, 739): "Malattie del sistema osteomuscolare e del tessuto connettivo",
    (740, 759): "Malformazioni congenite",
    (760, 779): "Alcune condizioni morbose di origine perinatale",
    (780, 799): "Sintomi, segni, e stati morbosi maldefiniti",
    (800, 999): "Traumatismi e avvelenamenti"
}

surgeries = {
    (0, 1): "PROCEDURE ED INTERVENTI NON CLASSIFICATI ALTROVE",
    (1, 5): "INTERVENTI SUL SISTEMA NERVOSO",
    (6, 7): "INTERVENTI SUL SISTEMA ENDOCRINO",
    (18, 20): "INTERVENTI SULL'ORECCHIO",
    (21, 29): "INTERVENTI SU NASO, BOCCA E FARINGE",
    (30, 34): "INTERVENTI SUL SISTEMA RESPIRATORIO",
    (35, 39): "INTERVENTI SUL SISTEMA CARDIOVASCOLARE",
    (40, 41): "INTERVENTI SUL SISTEMA EMATICO E LINFATICO",
    (42, 54): "INTERVENTI SULL’APPARATO DIGERENTE",
    (55, 59): "INTERVENTI SULL’APPARATO URINARIO",
    (60, 64): "INTERVENTI SUGLI ORGANI GENITALI MASCHILI",
    (65, 71): "INTERVENTI SUGLI ORGANI GENITALI FEMMINILI",
    (72, 75): "INTERVENTI OSTETRICI",
    (76, 84): "INTERVENTI SULL’APPARATO MUSCOLOSCHELETRICO",
    (85, 86): "INTERVENTI SUI TEGUMENTI",
    (87, 99): "MISCELLANEA DI PROCEDURE DIAGNOSTICHE E TERAPEUTICHE"
}

coloumn_to_save = ['Codice alfa numerico', 'Età', 'Sesso', 'Peso', 'Altezza', 'BMI', 'ASA', 'Mallampati',
                   'Cormack', 'Catetere vescicale', 'CVC', 'Diagnosi', 'Codice diagnosi', "Diabete Mellito", "Fumo",
                   "OSAS", "Pregressa polmonite(>30 gg)", "BPCO", "Ipertensione arteriosa",
                   "Cardiopatia ischemica cronica",
                   "Pregresso infarto miocardio", "Pregresso SCC", "Aritmie", "Ictus", "Pregresso TIA",
                   "Note_comorbidita", "Antipertensivi",
                   "Broncodilatatori", "Antiaritmici", "Anticoagulanti", "Antiaggreganti", "TIGO", "Insulina",
                   "Note_medicinali",
                   'Intervento', 'Codice intervento', 'Reintervento', 'Lateralità',
                   'Accesso chirurgico', 'Tempo Tot BO Ormaweb', 'Tempo Tot. SO OrmaWeb', 'Tempo Tot. RR',
                   'Numero chirurghi', 'Specializzando chirurgia', 'Numero anestesisti', 'Cambio anestesisti',
                   'Specializzando anestesia', 'Altro_comorbidita', "Altro_terapia"]


def common_data(list1, list2):
    """
    The function `common_data` checks if there is any common element between two lists and returns
    `True` if there is, otherwise it returns `False`.
    
    :param list1: The first list of elements to compare
    :param list2: The second list that you want to compare with the first list
    :return: a boolean value indicating whether there is any common element between the two input lists.
    """
    result = False

    # traverse in the 1st list
    for x in list1:

        # traverse in the 2nd list
        for y in list2:

            # if one common
            if x == y:
                result = True
                return result

    return result

def onehotencoding_forMultiLabelRow(dataset, column_name):
    """
    The function `onehotencoding_forMultiLabelRow` performs one-hot encoding on a specified column in a
    dataset, creating a new dataframe with binary values indicating the presence or absence of each
    unique value in the column.
    
    :param dataset: The dataset parameter is the input dataset that contains the rows and columns of
    data. It should be a pandas DataFrame object
    :param column_name: The column_name parameter is the name of the column in the dataset that you want
    to perform one-hot encoding on
    :return: a pandas DataFrame that contains the one-hot encoded values for the specified column in the
    dataset.
    """
    output_dict = {}
    column_index = dataset.columns.index(column_name)

    for index, row in enumerate(dataset.iter_rows()):
        values = list(row)  # Convert the tuple to a list of values
        column_values = values[column_index]

        for content in str(column_values).split(","):
            newcontent = content.strip()
            if newcontent != "" and newcontent:

                newcontent = unidecode.unidecode(newcontent)  # rimozione degli accenti

                substitution = None
                if column_name == "Note_medicinali":
                    substitution = terapia_pre_dict
                elif column_name == "Lateralità":
                    substitution = lateralit_pre_dict
                elif column_name == "Accesso chirurgico":
                    substitution = accesso_chirurgico_predict
                elif column_name == "Note_comorbidita":
                    substitution = comorbidita_dict
                elif column_name == "Mallampati":
                    substitution = mallampati_pre_dict
                elif column_name == "ASA":
                    substitution = asa_pre_dict
                elif column_name == "Diabete Mellito":
                    substitution = diabete_mellito_predict

                if substitution is not None:
                    for key, value in substitution.items():
                        if key in newcontent:
                            newcontent = newcontent.replace(key, value)

                if newcontent not in output_dict:
                    output_dict[newcontent] = [0] * len(dataset)
                    output_dict[newcontent][index] = 1

    #print("Type del output_dict", type(output_dict)) #questo print mi serve per capire se sto lavorando con i tipo giusti e non ci siano residui pandassiani : ) 
    output_df = pl.DataFrame(output_dict)
    #print("tipo output", (output_df)) #stampo l output per vedere se è giusto e sta effettivamente "rimependo"
    output_df = output_df.fill_nan(0)

    return output_df

def onehotencoding_forMultiLabelRow_forSURGERIES(dataset, column_name):
    output_df = pl.DataFrame()

    for i in range(len(dataset)):
        content = str(dataset[column_name][i]).split(",")[0]
        new_content = content.lstrip().rstrip().replace("V", "")

        if new_content != "" and new_content != "nan" and new_content:
            for key, value in surgeries.items():
                try:
                    # Check if new_content can be converted to a float
                    float_new_content = float(new_content)
                    if key[0] <= math.floor(float_new_content) <= key[1]:
                        to_add = value
                except ValueError:
                    try:
                        # new_content could not be converted to a float, so it's probably a date
                        date_time_obj = datetime.datetime.strptime(new_content, '%Y-%m-%d %H:%M:%S')
                        date_time_to_make_difference = datetime.datetime(1899, 12, 31, 0, 0, 0)
                        difference = date_time_obj - date_time_to_make_difference
                        new_code = (difference.total_seconds() // 3600) + (difference.total_seconds() - (
                                3600 * math.floor(difference.total_seconds() / 3600))) / (60 * 100)
                    except ValueError:
                        # new_content is a time, not a date
                        time_obj = datetime.datetime.strptime(new_content, '%H:%M:%S')
                        new_code = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second

                    # Convert new_code to a float and check if it's within the key range
                    float_new_code = float(new_code)
                    if key[0] <= math.floor(float_new_code) <= key[1]:
                        to_add = value

            # Update the Polars DataFrame with 1 in the specified column
            output_df = output_df.with_columns(pl.lit(1).alias(to_add))
            output_df = output_df.with_columns(pl.lit(1).alias("PROCEDURE_TOTALI"))

        else:
            # Update the Polars DataFrame with 1 in the specified column
            output_df = output_df.with_columns(pl.lit(1).alias("NO_INTERVENTO"))

    # Fill NaN values with 0
    #output_df = output_df.fill_nan(0)
    output_df = output_df.fill_nan(0)
    #print ("SSSSSSSSSSSSSSSSSSSSS",output_df)

    return output_df


BLE_Data = pl.read_csv("BLE_Data/raw_data.csv")
df = pd.read_excel("EHR_Data/BLOC-OP statistica.xlsx")


#^ COSA PIù IMPORTANTE DA FARE è TRASFORMARE TUTTO IN POLARS !!
#^ IN QUESTO MODO TUTTI GLI ERRORI SUCCESSIVI SARANNO DATI DA POLARS E NON DA PANDAS
#print ("Tipo del dataframe", type(df)) #sempre per contollare che non ci siano ancora traccie di pandas
for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)

# converto tutto in polars 
df_polars = pl.from_pandas(df)
#print ("CONTROLLO PER VEDERE SE EFFETTIVAMENTE SIA POLARS", type(df_polars))
data = pl.from_pandas(df[df.columns.intersection(coloumn_to_save)])
#print ("CONTROLLO MOLTO IMPORTAMTE ", type(data))

###################################################################################################################

# Puliamo un po' i dati elimando quelli che non hanno le features che ci servono
if not DATASET_GESTIONALE:
    data = data.filter((pl.col("Peso").is_not_null()) & (pl.col("Tempo Tot BO Ormaweb") != 0))
rooms = ["Sala_Operatoria_1", "Sala_Operatoria_2", "Sala_Operatoria_4"]

###################################################################################################################


#### Uniamo i dati anamnestici con i dati provenienti dall'architettura IOT ###
#### Cerchiamo anche di recuperare i pazienti che non hanno rilevazioni nella recovery room ####

# NEL CODICE ORIGINALE NON C'è UNA FUNZIONE MA UN FOR  CHE ACCEDE PER INDICE
#TUTTAVIA VISTO CHE IN POLARS NON POSSO ACCEDERE PER INDICE, HO FATTO PRIMA A CREARE UNA FUNZIONE CHE FA LA STESSA COSA
#E POI A RICHIAMARLA NEL FOR

def process_row0(row):
    """
    The function `process_row0` takes a row of data, performs calculations based on the values in the
    row, and returns the modified row.
    
    :param row: The parameter "row" is a polars DataFrame row that contains information about a specific
    item or entity. It is used to access and manipulate the data in that row
    :return: the modified "row" object.
    """
    ml_med_code = row["Codice alfa numerico"]

    if BLE_Data["identification_code"].eq(ml_med_code).any():
        current_id = BLE_Data.filter(BLE_Data["identification_code"].eq(ml_med_code))
        row = row.with_columns([pl.lit(list(current_id["feasible"])[0]).alias("feasible")])
        row = row.with_columns([pl.lit(round(sum(list(current_id["time_duration_minutes"])), 2)).alias("BLE_tot_BO_time")])

        for room in rooms:
            if room in current_id["room"].to_list():
                current_room = current_id.filter(current_id["room"].eq(room))
                row = row.with_columns([pl.lit(round(current_room["time_duration_minutes"].sum(), 2)).alias("BLE_tot_OR_time")])
                break
            else:
                row = row.with_columns([pl.lit(0).alias("BLE_tot_OR_time")])

        if "Recovery_Room" in current_id["room"].to_list():
            current_rr = current_id.filter(current_id["room"].eq("Recovery_Room"))
            row = row.with_columns([pl.lit(round(current_rr["time_duration_minutes"].sum(), 2)).alias("BLE_tot_RR_time")])
        else:
            if row["Tempo Tot. RR"][0] <= 10:
                row = row.with_columns([pl.lit(True).alias("feasible"), pl.lit(0).alias("BLE_tot_RR_time")])
            else:
                row = row.with_columns([pl.lit(0).alias("BLE_tot_RR_time")])
    else:
        row = row.with_columns(pl.lit(0).alias("BLE_tot_BO_time"))
        row = row.with_columns(pl.lit(0).alias("BLE_tot_RR_time"))
        row = row.with_columns(pl.lit(0).alias("BLE_tot_OR_time"))
        row = row.with_columns(pl.lit(False).alias("feasible"))
        
        
    row = row.with_columns([row["BLE_tot_OR_time"].cast(pl.Float64)])
    return row

# Creo un Polars DataFrame vuoto ma con le stesse colonne di data 
processed_data = pl.DataFrame({col: [] for col in data.columns})
is_first_row = True
# ITEROOOO
for index, row in enumerate(data.iter_rows()):
    row_dict = {col: [val] for col, val in zip(data.columns, row)}
    row_df = pl.DataFrame(row_dict)
    processed_row = process_row0(row_df)
    processed_row = processed_row.with_columns(processed_row["Codice alfa numerico"].cast(pl.datatypes.Utf8))
    processed_row = processed_row.with_columns(processed_row["BLE_tot_BO_time"].cast(pl.datatypes.Float64))
    processed_row = processed_row.with_columns(processed_row["BLE_tot_RR_time"].cast(pl.Float64))
    processed_row = processed_row.with_columns(processed_row["Catetere vescicale"].cast(pl.datatypes.Float64))
    processed_row = processed_row.with_columns(processed_row["CVC"].cast(pl.datatypes.Float64))
    processed_row = processed_row.with_columns(processed_row["Antipertensivi"].cast(pl.datatypes.Float64))
    processed_row = processed_row.with_columns(processed_row["Broncodilatatori"].cast(pl.datatypes.Float64))
    processed_row = processed_row.with_columns(processed_row["Antiaritmici"].cast(pl.datatypes.Float64))
    processed_row = processed_row.with_columns(processed_row["Anticoagulanti"].cast(pl.datatypes.Float64))
    processed_row = processed_row.with_columns(processed_row["Antiaggreganti"].cast(pl.datatypes.Float64))
    processed_row = processed_row.with_columns(processed_row["TIGO"].cast(pl.datatypes.Float64))
    processed_row = processed_row.with_columns(processed_row["Insulina"].cast(pl.datatypes.Float64))
    processed_row = processed_row.with_columns(processed_row["Altro_terapia"].cast(pl.datatypes.Float64))
    processed_row = processed_row.with_columns(processed_row["Cardiopatia ischemica cronica"].cast(pl.datatypes.Float64))
    processed_row = processed_row.with_columns(processed_row["Pregresso infarto miocardio"].cast(pl.datatypes.Float64))
    processed_row = processed_row.with_columns(processed_row["Pregresso SCC"].cast(pl.datatypes.Float64))
    processed_row = processed_row.with_columns(processed_row["Aritmie"].cast(pl.datatypes.Float64))
    processed_row = processed_row.with_columns(processed_row["Ictus"].cast(pl.datatypes.Float64))
    processed_row = processed_row.with_columns(processed_row["Pregresso TIA"].cast(pl.datatypes.Float64))
    processed_row = processed_row.with_columns(processed_row["Altro_comorbidita"].cast(pl.datatypes.Float64))
    processed_row = processed_row.with_columns(processed_row["Diabete Mellito"].cast(pl.datatypes.Float64))
    processed_row = processed_row.with_columns(processed_row["BPCO"].cast(pl.datatypes.Float64))
    processed_row = processed_row.with_columns(processed_row["Mallampati"].cast(pl.datatypes.Float64))
    processed_row = processed_row.with_columns(processed_row["Reintervento"].cast(pl.datatypes.Float64))
    processed_row = processed_row.with_columns(processed_row["Fumo"].cast(pl.datatypes.Float64))
    processed_row = processed_row.with_columns(processed_row["OSAS"].cast(pl.datatypes.Float64))
    processed_row = processed_row.with_columns(processed_row["Pregressa polmonite(>30 gg)"].cast(pl.datatypes.Float64))
    processed_row = processed_row.with_columns(processed_row["Ipertensione arteriosa"].cast(pl.datatypes.Float64))
    processed_row = processed_row.with_columns(processed_row["ASA"].cast(pl.datatypes.Float64))
    processed_row = processed_row.with_columns(processed_row["Numero chirurghi"].cast(pl.datatypes.Float64))
    

    # Se è la prima riga, inizializzo in processed_data con quella 
    if is_first_row:
        processed_data = processed_row
        is_first_row = False
    else:
        # Altrimenti faccio l"append" di polars della riga alla nuovo dataframe 
        processed_row = processed_row.select(processed_data.columns)
        #processed_row = processed_row.rename({'Codice alfa numerico': 'Codice alfa numerico' for old_name, new_name in zip(processed_row.columns, processed_data.columns)})
        processed_data = processed_data.vstack(processed_row)
        #print ("Come sono i dati processati? , processed_data) #cechk per vedere se va 


data = processed_data
#print("Processed data shape:", data)

###################################################################################################################
# riempimento dei dati nulli
# Fill missing values in the "ASA" and "Mallampati" columns
#controlo che la colonna fesible sia nel df, e se c'è mi faccio stampere che c'è altrimenti che non c'è

data = data.with_columns(pl.col("ASA").fill_nan(value=2))
data = data.with_columns(pl.col("Mallampati").fill_nan(value=1))

# List of columns to fill
lista_to_fill = ["Diabete Mellito", "Fumo", "OSAS", "Pregressa polmonite(>30 gg)", "BPCO", "Ipertensione arteriosa",
                 "Cardiopatia ischemica cronica", 
                 "Pregresso infarto miocardio", "Pregresso SCC", "Aritmie", "Ictus", "Pregresso TIA",
                 "Antipertensivi", "Broncodilatatori", "Antiaritmici", "Anticoagulanti", "Antiaggreganti", "TIGO",
                 "Insulina", "Altro_comorbidita", "Altro_terapia",
                 "CVC", "Reintervento", "Catetere vescicale"]

# Fill missing values in the columns in lista_to_fill
for elm in lista_to_fill:
   data = data.with_columns(pl.col(elm).fill_nan(value=0))

  
###################################################################################################################
### costruiamo anche un  dizionario che mappa che associa il codice intervento all'intervento stesso
codes = defaultdict()

for row in data.iter_rows():
    # Get the index of the "Codice intervento" column
    codice_intervento_index = data.columns.index("Codice intervento")


    if isinstance(row[codice_intervento_index], datetime.datetime):
        # devo contare quante ore e quanti minuti sono passati dal primo gennaio del 1900
        
        print ('SONO NELL IF DA CORREGGERE ')
        key = str(row[codice_intervento_index])

        date_time_obg = datetime.datetime.strptime(key, '%Y-%m-%d %H:%M:%S')
        date_time_to_make_difference = datetime.datetime(1899, 12, 31, 0, 0, 0)

        difference = date_time_obg - date_time_to_make_difference

        key = (difference.total_seconds() // 3600) + (
                difference.total_seconds() - (3600 * math.floor(difference.total_seconds() / 3600))) / (60 * 100)

        data = data.with_column("Codice intervento", when(col("Codice intervento") == row[codice_intervento_index])
            .then(key)
            .otherwise(col("Codice intervento")))

    elif isinstance(row[codice_intervento_index], datetime.time):
        
        print("sono nell elfi")
        key = str(row[codice_intervento_index])
        date_time_obg = datetime.datetime.strptime(key, '%H:%M:%S').time()

        key = float(str(date_time_obg.hour) + "." + str(date_time_obg.minute))

        data = data.with_column("Codice intervento", when(col("Codice intervento") == row[codice_intervento_index])
            .then(key)
            .otherwise(col("Codice intervento")))


###################################################################################################################

OH_encoded_Column = ["ASA", "Mallampati", "Accesso chirurgico", "Lateralità", "Diabete Mellito"]

for column in OH_encoded_Column:
    df_encoded = onehotencoding_forMultiLabelRow(data, column)
    #questi print ci sono anche nel codice di partenza 
    print ("                       ",type(df_encoded))
    print ("                       ",type(data))
    if 'None' in df_encoded.columns:
        df_encoded = df_encoded.rename({'None': f'{column}_None'})
    data = data.hstack(df_encoded)

#print("Data shape after one-hot encoding:", data)

################################################

df_encoded_surgeries = onehotencoding_forMultiLabelRow_forSURGERIES(data, "Codice intervento")

df_encoded_surgeries = df_encoded_surgeries.with_columns(pl.arange(0, df_encoded_surgeries.shape[0]).alias("index"))


missing_rows = data.shape[0] - df_encoded_surgeries.shape[0]
if missing_rows > 0:
    for _ in range(missing_rows):
        df_encoded_surgeries = df_encoded_surgeries.vstack(df_encoded_surgeries.tail(1))

data = data.hstack(df_encoded_surgeries.drop("index"))

#print("data dopo", data)
###################################################################################################################

# dobbiamo aggiornare i vari valori del codice intervento perché alcuni sono formattati come dati o orari
# Codice per contare le tipologie di intervento presenti nel database, sarà utile in future per il bilanciamento
# del dataset
# data.drop(data[data["Codice intervento"] == "nan"].index, inplace=True)
count = {}

# Supponendo che 'surgeries' sia un dizionario di chiavi e valori
for value in surgeries.values():
    if value in data.columns:
        count[value] = (data[value] == 1).sum()


###################################################################################################################
colonne_da_rimuovere = ["Intervento", "Codice intervento", "Accesso chirurgico", 
                        "Codice diagnosi", "Diagnosi", "Note_comorbidita", 
                        "Note_medicinali", "Cormack", "Mallampati", "ASA", 
                        "Diabete Mellito", "NO_DIABETE_MELLITO"]

# Rimuovi le colonne specificate
data = data.drop(colonne_da_rimuovere)


def process_row(row):
    """
    The function `process_row` checks if the first element of a row is null, and if so, it checks if the
    second and third elements are also null, and if not, it returns the sum of the second and third
    elements; otherwise, if the first element is not null, it returns the first element.
    
    :param row: The parameter "row" is a list or tuple containing three elements. The first element is
    expected to be a value, while the second and third elements can be either values or None
    :return: the value of row[0] if it is not null. If row[0] is null, it checks if row[1] and row[2]
    are also null. If they are null and DATASET_GESTIONALE is empty, it returns None. Otherwise, it
    returns the sum of row[1] and row[2].
    """
    if row[0].is_null():
        if row[1].is_null() and row[2].is_null():
            if not DATASET_GESTIONALE:
                return None
        else:
            return row[1] + row[2]
    else:
        return row[0]

data = data.with_columns([
    pl.when(pl.col("Tempo Tot BO Ormaweb").is_null() & 
            pl.col("Tempo Tot. SO OrmaWeb").is_null() & 
            pl.col("Tempo Tot. RR").is_null() & 
            (DATASET_GESTIONALE == 0)).then(None)
    .when(pl.col("Tempo Tot BO Ormaweb").is_null()).then(pl.col("Tempo Tot. SO OrmaWeb") + pl.col("Tempo Tot. RR"))
    .otherwise(pl.col("Tempo Tot BO Ormaweb")).alias("Tempo Tot BO Ormaweb")
])

# Add the "BMI" column
data = data.with_columns("BMI", pl.col("Peso") / (pl.col("Altezza") ** 2))


print (data)

####################################################à



if not DATASET_GESTIONALE:
    #print('CIAOOOOOOOOOOOOOOOOOOOOOO')
    data = data.drop_nulls()

data = data.drop("#N/D")
data = data.drop("0.0")

print(data)


#####################FINO QUI FATTO , DA SOTTO è ANCORA PANDAS ##################
##^28 12 

data = data.filter(pl.col("INTERVENTI SULL’APPARATO DIGERENTE") == 1)

to_drop = ['INTERVENTI SUL SISTEMA ENDOCRINO', 'INTERVENTI SULL’APPARATO URINARIO',
           'INTERVENTI SULL’APPARATO MUSCOLOSCHELETRICO',
           'INTERVENTI SUL SISTEMA EMATICO E LINFATICO',
           'INTERVENTI SUI TEGUMENTI',
           'INTERVENTI SUGLI ORGANI GENITALI FEMMINILI',
           'INTERVENTI SUL SISTEMA RESPIRATORIO',
           'INTERVENTI SUL SISTEMA CARDIOVASCOLARE',
           'INTERVENTI SUGLI ORGANI GENITALI MASCHILI',
           'MISCELLANEA DI PROCEDURE DIAGNOSTICHE E TERAPEUTICHE']

for elmnt in to_drop:
    if elmnt in data.columns:
        data = data.drop(elmnt)
#print('WIIIIIIIIIIIIIIIIIIIIIIIIIII', data.columns)
print ("dopo drop", data)


##FIN SOPRA DOVREBBE ESSERE TUTTO OK #####

import random
import math

if not DATASET_GESTIONALE:
    for i in tqdm(range(len(data))):  # iterate over rows
        for j in range(len(data.columns)):  # iterate over columns
            value = data[i, j]  # get cell value
            try:
                if value is None:
                    print(i, j)
            except:
                pass

    index_for_validation = []
    for key in count:
        occurence = count[key]
        if key in data.columns:
            if occurence >= 6:
                to_take = math.ceil(0.05 * occurence)

                list_index = data.filter(pl.col(key) == 1).select(pl.col("*")).to_pandas().index.tolist()
                random_extracted = random.sample(list_index, k=to_take)
                index_for_validation.extend(random_extracted)
            else:
                data = data.filter(pl.col(key) != 1)
                data = data.drop(key)

    index_for_validation = list(set(index_for_validation))

    index_for_test = []
    for key in count:
        occurence = count[key]
        if occurence >= 4:
            to_take = math.ceil(0.1 * occurence)
            if key in data.columns:
                list_index = data.filter(pl.col(key) == 1).select(pl.col("*")).to_pandas().index.tolist()
                random_extracted = random.sample(list_index, k=to_take)
                while not common_data(index_for_validation, random_extracted):
                    random_extracted = random.sample(list_index, k=to_take)
                index_for_test.extend(random_extracted)

    index_for_test = list(set(index_for_test))

    # Assuming `codes` is a dictionary mapping old column names to new column names
    data = data.rename(codes)
    
    
    for old_name, new_name in codes.items():
        data = data.with_column_renamed(old_name, new_name)

    #data = data.with_column_renamed(codes.keys(), codes.values())
    
    
    
    print("Numero di colonne:", len(list(data.columns)))
    print("dataaaaaaaaaaaaaaaaaaaaaaaaa",data)

###########################################################################################################
# Prima di salvare il tutto dobbiamo riempire automaticamente i campi vuoti del Bluetooth che sono mancanti
# Prima riempiamo tutti i campi che sono a 0
# Poi controlliamo che: se la recovery in BL è 0 e in ormaweb è <10 allora non facciamo nulla, se il tempo BO del bluetooth è minore della somma di SO + RR dopo la sostituzione allora
# sostituiamo anche lui con i dati di ormaweb


## !da qui, tutto quello che c'è sopra dovrebbe andare bene : ) speriamo cacchio 

condition = data["feasible"] == False

data = data.with_columns(
    "BLE_tot_BO_time", pl.when(condition).then(data["Tempo Tot BO Ormaweb"]).otherwise(data["BLE_tot_BO_time"])
)

data = data.with_columns(
    "BLE_tot_RR_time", pl.when(condition).then(data["Tempo Tot. RR"]).otherwise(data["BLE_tot_RR_time"])
)

data = data.with_columns(
    "BLE_tot_OR_time", pl.when(condition).then(data["Tempo Tot. SO OrmaWeb"]).otherwise(data["BLE_tot_OR_time"])
)



data = data.filter(data["feasible"])



# Update "BLE_tot_OR_time" where it's 0 with "Tempo Tot. SO OrmaWeb"
data = data.with_columns(
    pl.when(data["BLE_tot_OR_time"] == 0)
    .then(data["Tempo Tot. SO OrmaWeb"])
    .otherwise(data["BLE_tot_OR_time"])
    .alias("BLE_tot_OR_time")
)



# Update "BLE_tot_RR_time" where it's 0 and "Tempo Tot. RR" > 10 with "Tempo Tot. RR"
data = data.with_columns(
    pl.when((data["BLE_tot_RR_time"] == 0) & (data["Tempo Tot. RR"] > 10))
    .then(data["Tempo Tot. RR"])
    .otherwise(data["BLE_tot_RR_time"])
    .alias("BLE_tot_RR_time")
)

# Update "BLE_tot_BO_time" where it's 0 with "Tempo Tot BO Ormaweb"
data = data.with_columns(
    pl.when(data["BLE_tot_BO_time"] == 0)
    .then(data["Tempo Tot BO Ormaweb"])
    .otherwise(data["BLE_tot_BO_time"])
    .alias("BLE_tot_BO_time")
)



# Update "BLE_tot_BO_time" where "BLE_tot_OR_time" + "BLE_tot_RR_time" > "BLE_tot_BO_time" with "Tempo Tot BO Ormaweb"
data = data.with_columns(
    pl.when((data["BLE_tot_OR_time"] + data["BLE_tot_RR_time"]) > data["BLE_tot_BO_time"])
    .then(data["Tempo Tot BO Ormaweb"])
    .otherwise(data["BLE_tot_BO_time"])
    .alias("BLE_tot_BO_time")
)



##aggiungiamo le sale e tutte le colonne che vuole laura

if DATASET_GESTIONALE:
    # Join the two DataFrames on "Codice alfa numerico"
    data = data.join(df, on="Codice alfa numerico", how="left")

    # Rename the columns from df
    data = data.with_columns([
        pl.col("Sala").alias("Sala"),
        pl.col("Ingresso BO OrmaWeb").alias("Ingresso BO OrmaWeb"),
        pl.col("Sala pronta OrmaWeb").alias("Sala pronta OrmaWeb"),
        pl.col("Ingresso SO OrmaWeb").alias("Ingresso SO OrmaWeb"),
        pl.col("Inizio Anestesia OrmaWeb").alias("Inizio Anestesia OrmaWeb"),
        pl.col("Inzio Chirurgia OrmaWeb").alias("Inzio Chirurgia OrmaWeb"),
        pl.col("Fine Chir. OrmaWeb").alias("Fine Chir. OrmaWeb"),
        pl.col("Fine Anest. OrmaWeb").alias("Fine Anest. OrmaWeb"),
        pl.col("Uscita SO OrmaWeb").alias("Uscita SO OrmaWeb"),
        pl.col("Uscita BO OrmaWeb").alias("Uscita BO OrmaWeb"),
        pl.col("Ripristino SO OrmaWeb").alias("Ripristino SO OrmaWeb")
    ])


data = data.drop("INTERVENTI SULL’APPARATO DIGERENTE")

data.write_csv("MLMED_Dataset.csv")
data.to_pandas().to_excel("MLMED_Dataset.xlsx", index=False)


from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

if not DATASET_GESTIONALE:
    """ validation = data.clone()
    test = data.clone()

    validation = validation.filter(pl.col('Codice alfa numerico').cast(pl.Utf8).is_in(index_for_validation))
    test = test.filter(pl.col('Codice alfa numerico').cast(pl.Utf8).is_in(index_for_test))
    data = data.filter(~pl.col('Codice alfa numerico').cast(pl.Utf8).is_in(index_for_validation + index_for_test))

    data.write_csv("MLMED_Dataset_train.csv")
    validation.write_csv("ML_MED_Dataset_validation.csv")
    test.write_csv("ML_MED_Dataset_test.csv") """
    

    data = pl.read_csv("MLMED_Dataset.csv", null_values=["#RIF!"])

    # Split the data into training and remaining data (for validation and test)
    train, remaining = train_test_split(data, test_size=0.3, random_state=42)

    # Split the remaining data into validation and test sets
    validation, test = train_test_split(remaining, test_size=2/3, random_state=42)
    
    train.write_csv("MLMED_Dataset_train.csv")
    validation.write_csv("ML_MED_Dataset_validation.csv")
    test.write_csv("ML_MED_Dataset_test.csv")
    
    
    
