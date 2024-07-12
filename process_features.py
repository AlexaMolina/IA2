from descriptor import glcm, bitdesc
import numpy as np
import os

def process_datasets(root_folder):
    all_features_glcm = []
    all_features_bitdesc = []
    #count = 1
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('.jpeg', '.jpg', '.png', '.bmp', '.tiff')):
                #camino relativo sin la raiz
                #relative_path = os.path.realpath(os.path.join(root,file),root_folder)
                #camino relativo con la raiz
                image_rel_path = os.path.join(root,file)
                # nombre de la carpeta donde esta la imagen
                folder_name = os.path.basename(os.path.dirname(image_rel_path))
                #extraccion glcm
                extraction_glcm = glcm(image_rel_path)
                extraction_glcm = extraction_glcm + [folder_name, image_rel_path]
                all_features_glcm.append(extraction_glcm)
                #extraccion bitdesc
                extraction_bitdesc = bitdesc(image_rel_path)
                extraction_bitdesc = extraction_bitdesc + [folder_name, image_rel_path]
                all_features_bitdesc.append(extraction_bitdesc)
                
    print('Extraction Complete!')
    #signature glcm
    signatures_glcm = np.array(all_features_glcm)
    np.save('features_glcm.npy',signatures_glcm)
    print('Signature Glcm saved:', len(signatures_glcm))
    #signature bitdesc
    signatures_bitdesc = np.array(all_features_bitdesc)
    np.save('features_bitdesc.npy',signatures_bitdesc)
    print('Signature Bitdesc saved:', len(signatures_bitdesc))
#carpeta de la las imagenes
process_datasets('images/')