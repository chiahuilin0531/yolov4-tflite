# cat pse real > aug
#########  train  #############
# 1 cls 
cat ../Pse_data/anno/train_pse_1cls.txt ../real_data/anno/train_real_1cls.txt > train_mix_1cls.txt 
# 3 cls
cat ../Pse_data/anno/train_pse_3cls.txt ../real_data/anno/train_real_1cls.txt > train_mix_3cls.txt
# 4 cls 
cat ../Pse_data/anno/train_pse_4cls.txt ../real_data/anno/train_real_1cls.txt > train_mix_4cls.txt
##########  val  ##############
# 1 cls
cat ../Pse_data/anno/val_pse_1cls.txt  ../real_data/anno/val_real_1cls.txt > val_mix_1cls.txt
# 3 cls
cat ../Pse_data/anno/val_pse_3cls.txt  ../real_data/anno/val_real_1cls.txt > val_mix_3cls.txt
# 4 cls
cat ../Pse_data/anno/val_pse_4cls.txt  ../real_data/anno/val_real_1cls.txt > val_mix_4cls.txt
###############################
