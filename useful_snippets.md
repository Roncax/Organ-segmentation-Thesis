# Databases
SegThor:
"0": "Bg",
"1": "Esophagus",
"2": "Heart",
"3": "Trachea",
"4": "Aorta"

Structseg:
"0": "Bg",
"1": "RightLung",
"2": "LeftLung",
"3": "Heart",
"4": "Trachea",
"5": "Esophagus",
"6": "SpinalCord"


DBs: SegTHOR, StructSeg2019_Task3_Thoracic_OAR

# Various
sources: local, gradient, polimi
Models: seresunet, unet, segnet, deeplabv3

What to change in Gradient:
source in training:     /notebooks/Organ-segmentation-Thesis
platform: "gradient"

Losses: dice, focal, crossentropy, dc_ce, twersky, jaccard