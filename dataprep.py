import os

"""
script to process all data for network preprocessing, one by one.
input by user for RF Familiy, no loop for all, to avoid errors
"""
#---------------RF Familiy definition-------------------
print("Enter familiy name like RF00034") # dummy: RF02503_dummy
#Familiy = input()
#Families = os.listdir("/home/mgarcia/si_RNA_Network/source/Syntney_out_new/")
Families = [input()]
#--------------------env-commands----------------------
Syntney_env="conda activate Syntney"
dataprocessing_env="conda activate dataprocessing"

for Familiy in Families:
    # ----------------------pathes--------------------------
    root = "/home/mgarcia/si_RNA_Network/source/"
    Syntney_folder = root + "Syntney_out_new/"
    glassgo_file = root + "glassgo/" + Familiy + ".glassgo"
    Syntney_database = "-d /home/mgarcia/media/cyano_share/exchange/Mario/synt_rRNA_fayyaz.db "

    fasta_file = Syntney_folder + Familiy + "/_Network.fasta"
    cm_model = root + "cm_models/" + Familiy + ".cm"
    ResCM_file = root + "cm_out/" + Familiy + ".ResCM"
    tsv_file_out = root + "tsv_files_new/" + Familiy + ".tsv"

    # ------Syntney/clustalo/RNApdist out-path----------------
    Syntney_clustalo_out = Syntney_folder + Familiy + "/"
    # --------Syntney-----------
    Syntney_start = "python /home/mgarcia/si_RNA_Network/Syntney/Syntney.py "
    Syntney_propperties = " -x 30 -n cys -r off --node_normalization True "
    Syntney = Syntney_start + "-i " + glassgo_file + Syntney_propperties + Syntney_database + "-o " + Syntney_clustalo_out
    # --------Infernal---------
    infernal_cpu = "--cpu 30 "
    cm_calibrate = "cmcalibrate " + infernal_cpu + cm_model
    cm_press = "cmpress -F " + cm_model
    cm_search_propperties = "--noali --toponly " + infernal_cpu
    infernal_out = "--tblout " + root + "cm_out/" + Familiy + ".ResCM "
    cm_search = "cmsearch " + cm_search_propperties + infernal_out + cm_model + " " + fasta_file
    # ---------clustalo-----------
    clustalo = "clustalo -i " + fasta_file + " --force --distmat-out " + Syntney_clustalo_out + "PID.txt --full --percent-id --threads=30"
    #-------------------RNApdist---------------------------------------------
    RNApdist_outfile = Syntney_clustalo_out+"pdist_values.pdist"
    RNApdist = "RNApdist < "+fasta_file+" -Xm > "+RNApdist_outfile
    #-----------------controle section------------------------
    Syntney_file_list = os.listdir(Syntney_clustalo_out)
    Syntney_Flag = False
    ResCM_Flag = False
    tsv_Flag = False
    clustalo_Flag = False
    RNApdist_Flag = False
    for i in Syntney_file_list:
        if i == "_Network.fasta":
            print("Syntney done")
            Syntney_Flag = True
        elif i == "PID.txt":
            print("clustalo done")
            clustalo_Flag = True
        elif i == "pdist_values.pdist":
            print("RNApdist done")
            RNApdist_Flag = True
    infernal_out_check_path = root+"cm_out/"
    ResCM_file_list = os.listdir(infernal_out_check_path)
    ResCM_check = Familiy+".ResCM"
    for i in ResCM_file_list:
        if i == ResCM_check:
            print("Infernal done")
            ResCM_Flag = True
    tsv_folder_path = root+"tsv_files_new/"
    tsv_file = Familiy+".tsv"
    tsv_list = os.listdir(tsv_folder_path)
    for i in tsv_list:
        if i == tsv_file:
            print("tsv preparation done")
            tsv_Flag = True
    #-----------------operation sektion------------------------------------------
    #os.system(Syntney_env)
    if Syntney_Flag == False:
        print(Syntney)
        os.system(Syntney)
    #os.system(dataprocessing_env)
    if ResCM_Flag == False:
        print(cm_calibrate)
        os.system(cm_calibrate)
        print(cm_press)
        os.system(cm_press)
        print(cm_search)
        os.system(cm_search)
    if tsv_Flag == False:
        print("do tsv")
        with open(ResCM_file, 'r') as a_file:
            with open(tsv_file_out, 'w') as f:
                for i in a_file:
                    i = i.strip('\n')
                    if not '#' in i:
                        body = []
                        c = i.split('  ')
                        for t in c:
                            u = ''
                            if t != u:
                                body.append(t)
                        name = body[0].split(' ')[0]
                        eval = body[-2].split(' ')
                        if '' in eval:
                            eval.remove('')
                        f.write(name + '\t')
                        f.write(eval[0] + '\t')
                        f.write(eval[1] + '\t')
                        f.write(body[-1] + '\n')
    if clustalo_Flag == False:
        print(clustalo)
        os.system(clustalo)
    if RNApdist_Flag == False:
        print(RNApdist)
        os.system(RNApdist)